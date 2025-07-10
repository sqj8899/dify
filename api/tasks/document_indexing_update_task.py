import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore

from core.indexing_runner import DocumentIsPausedError, IndexingRunner
from core.rag.index_processor.index_processor_factory import IndexProcessorFactory
from extensions.ext_database import db
from models.dataset import Dataset, Document, DocumentSegment


@shared_task(queue="dataset")
def document_indexing_update_task(dataset_id: str, document_id: str):
    """
    Async update document
    :param dataset_id:
    :param document_id:

    Usage: document_indexing_update_task.delay(dataset_id, document_id)
    """
    logging.info(click.style("Start update document: {}".format(document_id), fg="green"))
    start_at = time.perf_counter()

    document = db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()

    if not document:
        logging.info(click.style("Document not found: {}".format(document_id), fg="red"))
        db.session.close()
        return

    document.indexing_status = "parsing"
    document.processing_started_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    db.session.commit()

    # delete document segments except manual ones
    try:
        dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise Exception("Dataset not found")

        index_type = document.doc_form
        index_processor = IndexProcessorFactory(index_type).init_index_processor()

        # 获取所有segment，保留手动创建的segment数据库记录，但所有向量索引都要删除重新embedding
        segments = db.session.query(DocumentSegment).filter(DocumentSegment.document_id == document_id).all()
        if segments:
            # 分离手动创建的segment（is_manual_created为True）和其他segment
            manual_segments = [segment for segment in segments if segment.is_manual_created is True]
            auto_segments = [segment for segment in segments if segment.is_manual_created is not True]

            # 删除所有segment的向量索引（包括手动创建的）
            all_index_node_ids = [segment.index_node_id for segment in segments]
            index_processor.clean(dataset, all_index_node_ids, with_keywords=True, delete_child_chunks=True)

            # 只删除自动创建的segment数据库记录
            if auto_segments:
                for segment in auto_segments:
                    db.session.delete(segment)
                db.session.commit()

            # 将保留的手动创建的segment状态设置为"indexing"，以便重新创建向量索引
            if manual_segments:
                for segment in manual_segments:
                    segment.status = "indexing"
                    segment.indexing_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                db.session.commit()

            logging.info(
                click.style(
                    "Deleted {} auto segments, preserved {} manual segments for document: {}".format(
                        len(auto_segments), len(manual_segments), document_id
                    ),
                    fg="green",
                )
            )
            logging.info(
                click.style(
                    "Deleted all vector indexes for {} total segments (will be re-embedded)".format(
                        len(segments)
                    ),
                    fg="green",
                )
            )
        end_at = time.perf_counter()
        logging.info(
            click.style(
                "Cleaned document when document update data source or process rule: {} latency: {}".format(
                    document_id, end_at - start_at
                ),
                fg="green",
            )
        )
    except Exception:
        logging.exception("Cleaned document when document update data source or process rule failed")

    try:
        indexing_runner = IndexingRunner()
        # 使用run_in_indexing_status方法，只对现有的segment重新创建向量索引
        # 而不是重新从原始文件分割文档（这会覆盖手动创建的segment）
        indexing_runner.run_in_indexing_status(document)
        end_at = time.perf_counter()
        logging.info(click.style("update document: {} latency: {}".format(document.id, end_at - start_at), fg="green"))
    except DocumentIsPausedError as ex:
        logging.info(click.style(str(ex), fg="yellow"))
    except Exception:
        logging.exception("document_indexing_update_task failed, document_id: {}".format(document_id))
    finally:
        db.session.close()
