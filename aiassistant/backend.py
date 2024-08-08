import ast
import os
import time
from pathlib import Path

import requests
from dynaconf import settings
from slugify import slugify
from nicegui import run
from aiassistant.core import update_audit
from aiassistant.database import add_upload_stats, tb_parse_stats, tb_upload_stats, tb_qa_stats
from aiassistant.log import get_logger

media_dir = settings.UPLOADS_DIR

logger = get_logger(__name__)


def process_uploads(name, file_content):
    Path(media_dir).mkdir(exist_ok=True)
    file_id = int(time.strftime('%Y%m%d%H%M%S', time.gmtime()))
    file_name = str(file_id) + '-' + slugify(Path(name).stem) + '.pdf'
    file_media_path = Path(media_dir) / file_name
    update_audit(form_id=file_id, task='UPLOAD')
    with open(file_media_path, 'wb') as file_obj:
        logger.info(f'Writing file at {str(file_media_path)}')
        file_obj.write(file_content)
        logger.info(f'Uploaded file written to {str(file_media_path)}')
    file_size = os.path.getsize(file_media_path)
    upload_stat = dict(
        name=name,
        file_name=file_name,
        file_size=file_size,
        status=settings.FORM_STATUS_UPLOADED
    )
    add_upload_stats(upload_stat=upload_stat, file_id=file_id)
    update_audit(form_id=file_id, task='UPLOAD')
    return file_id


async def process_parse_form(form_id: int):
    parse_form_url = f'http://localhost:8080/api/parse_form/{form_id}'
    task, code, error_msg, tags = 'PARSE', 'SUCCESS', None, f'model_parse={settings.MODEL_PARSE}'
    logger.info(f'Starting form parsing for form_id: {form_id}')
    update_audit(form_id=form_id, task=task, tags=tags)
    try:
        response = await run.io_bound(requests.get, parse_form_url, timeout=900)
        response.raise_for_status()
        form_fields = ast.literal_eval(response.json()['form_fields'])
        parse_form_docs = [dict(form_id=form_id, field=x) for x in form_fields]
        tb_parse_stats.insert_multiple(parse_form_docs)
        tb_upload_stats.update(dict(status=settings.FORM_STATUS_PARSED), doc_ids=[form_id])
    except Exception as error:
        code, error_msg = 'ERROR', str(error)
        logger.exception('Error during Parse stage.')
    finally:
        update_audit(form_id=form_id, task=task, tags=tags, code=code, error_msg=error_msg)
    logger.info(f'Completed form parsing for form_id: {form_id}')


async def process_qa_form(form_id: int):
    qa_form_url = f'http://localhost:8080/api/qa_form/{form_id}'
    logger.info(f'Starting form QA for form_id: {form_id}')
    task, code, error_msg, tags = 'AUTO_QA', 'SUCCESS', None, f'model_qa={settings.MODEL_QA}'
    update_audit(form_id=form_id, task=task, tags=tags)
    try:
        response = await run.io_bound(requests.get, qa_form_url, timeout=900)
        response.raise_for_status()
        qa_response = response.json()
        qa_stats = qa_response['qa_stats']
        logger.info(f'Adding {len(qa_stats)} qa_stats to DB')
        qa_form_docs = [{**dict(form_id=form_id), **qa_stat} for qa_stat in qa_stats]
        tb_qa_stats.insert_multiple(qa_form_docs)
        tb_upload_stats.update(dict(status=settings.FORM_STATUS_QA), doc_ids=[form_id])
    except Exception as error:
        code, error_msg = 'ERROR', str(error)
        logger.exception('Error during QA stage.')
    finally:
        update_audit(form_id=form_id, task=task, tags=tags, code=code, error_msg=error_msg)
    logger.info(f'Completed form QA for form_id: {form_id}.')