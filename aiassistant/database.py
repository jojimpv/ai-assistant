"""
Module to host TinyDB operations.
The database operation like insert, update and query to different DB tables are grouped under this script.
"""

from dynaconf import settings
from tinydb import TinyDB, Query
from tinydb.table import Document

from aiassistant.log import get_logger

db = TinyDB(settings.DB_PATH)
tb_upload_stats = db.table('upload_stats')
tb_parse_stats = db.table('parse_stats')
tb_qa_stats = db.table('qa_stats')
tb_audit = db.table('audit')

logger = get_logger(__name__)


def add_upload_stats(upload_stat: dict, file_id: int):
    tb_upload_stats.insert(Document(upload_stat, doc_id=file_id))


def get_upload_stats(form_id=None):
    if form_id:
        upload_stat = tb_upload_stats.get(doc_id=form_id)
        return upload_stat
    upload_stats = tb_parse_stats.all()
    return upload_stats


def get_parse_stats(form_id, parse_id=None):
    if parse_id:
        parse_stat = tb_parse_stats.get(doc_id=parse_id)
        return parse_stat
    ParseStat = Query()
    parse_stats = tb_parse_stats.search(ParseStat.form_id == form_id)
    return parse_stats


def get_qa_stats(form_id, parse_id=None):
    Item = Query()
    if parse_id:
        qa_stat = tb_qa_stats.search((Item.form_id == form_id) & (Item.question_id == parse_id))[0]
        return qa_stat
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    return qa_stats


def get_first_question_answer(form_id):
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    first_qa = qa_stats[0]
    return first_qa


def get_next_question_answer(form_id, index=0):
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    qa = qa_stats[index]
    question_id = qa.get('question_id')
    question = get_parse_stats(form_id, parse_id=question_id)['field']
    result = {**qa, **dict(question=question), **dict(question_id=question_id)}
    return result


def get_combined_qa(form_id):
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    results = []
    for qa in qa_stats:
        question_id = qa.get('question_id')
        question = get_parse_stats(form_id, parse_id=question_id)['field']
        answer_auto = qa.get('answer_auto', '')
        answer_user = qa.get('answer_user', '')
        if answer_user:
            answer = answer_user
            source = 'user'
        else:
            answer = answer_auto
            source = 'auto'

        result = dict(question=question, answer=answer, source=source, answer_auto=answer_auto, answer_user=answer_user)
        results.append(result)
    return results


def get_max_question_answer(form_id):
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    qa_len = len(qa_stats)
    return qa_len


def get_qa_doc_id_from_question_id(question_id: int):
    QaStat = Query()
    doc_id = tb_qa_stats.search(QaStat.question_id == question_id)[0].doc_id
    return doc_id


def save_user_answer(question_id, answer):
    doc_id = get_qa_doc_id_from_question_id(question_id=question_id)
    tb_qa_stats.update(dict(answer_user=answer), doc_ids=[doc_id])
    logger.info(f'Updated {question_id= } with {answer=}')


def get_all_qa():
    all_qa_docs = tb_qa_stats.all()
    return all_qa_docs


def get_audit_record(event='form_process', form_id=None, task=None, sub_task=None):
    Audit = Query()
    audit_recs = tb_audit.search(
        (Audit.event == event) & (Audit.form_id == form_id) & (Audit.task == task) & (Audit.sub_task == sub_task))
    return audit_recs[-1] if audit_recs else None


def upsert_audit_record(delta: dict, doc_id=None):
    # logger.info(f'upsert_audit_record with {doc_id=}, {delta=}')
    if doc_id:
        tb_audit.update(delta, doc_ids=[doc_id])
    else:
        tb_audit.insert(delta)


def get_all_audit(form_id=None):
    if form_id:
        Audit = Query()
        audit_records = tb_audit.search(Audit.form_id == form_id)
    else:
        audit_records = tb_audit.all()
    return sorted(audit_records, key=lambda x: x['start_time'], reverse=True)
