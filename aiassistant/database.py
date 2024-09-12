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


def add_upload_stats(upload_stat: dict, file_id: int) -> None:
    """Insert upload stats to persistent DB.

    Args:
        upload_stat: Upload stats
        file_id: Form ID

    Returns:
        None

    """
    tb_upload_stats.insert(Document(upload_stat, doc_id=file_id))


def get_upload_stats(form_id: int = None) -> Document | list[Document]:
    """Get all or specific form_id's upload stats

    Args:
        form_id: Form ID

    Returns:
        Upload starts for given form_id or all upload stats if form_id is None

    """
    if form_id:
        upload_stat = tb_upload_stats.get(doc_id=form_id)
        return upload_stat
    upload_stats = tb_parse_stats.all()
    return upload_stats


def get_parse_stats(form_id: int, parse_id: int = None) -> list[Document]:
    """Get parse stats for given parse_id or all for given form_id

    Args:
        form_id: Form ID
        parse_id: Parse ID

    Returns:
        Parse stats for given parse_id or all for given form_id

    """
    if parse_id:
        parse_stat = tb_parse_stats.get(doc_id=parse_id)
        return parse_stat
    ParseStat = Query()
    parse_stats = tb_parse_stats.search(ParseStat.form_id == form_id)
    return parse_stats


def get_qa_stats(form_id: int, parse_id: int = None) -> list[Document]:
    """Get QA stats for given parse_id or all for given form_id

    Args:
        form_id: Form ID
        parse_id: Parse ID

    Returns:
        QA stats for given parse_id or all for given form_id

    """
    Item = Query()
    if parse_id:
        qa_stat = tb_qa_stats.search((Item.form_id == form_id) & (Item.question_id == parse_id))[0]
        return qa_stat
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    return qa_stats


def get_first_question_answer(form_id: int):
    """Get first QA for given form_id

    Args:
        form_id: Form ID

    Returns:
        First QA for given form_id

    """
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    first_qa = qa_stats[0]
    return first_qa


def get_next_question_answer(form_id: int, index: int = 0) -> dict:
    """Get QA stats for the given form_id at index given

    Args:
        form_id: Form ID
        index: Index

    Returns:
        QA stats for the given form_id at index given

    """
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    qa = qa_stats[index]
    question_id = qa.get('question_id')
    question = get_parse_stats(form_id, parse_id=question_id)['field']
    result = {**qa, **dict(question=question), **dict(question_id=question_id)}
    return result


def get_combined_qa(form_id: int) -> list[dict]:
    """Combine `answer_auto` and `answer_user` by giving preference to `answer_user` in QA

    Args:
        form_id: Form ID

    Returns:
        List of combined QA stats with question, answer and source attributes

    """
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


def get_max_question_answer(form_id: int) -> int:
    """Get number of QA available for given form_id

    Args:
        form_id: Form ID

    Returns:
        Number of QA available for given form_id

    """
    QaStat = Query()
    qa_stats = tb_qa_stats.search(QaStat.form_id == form_id)
    qa_len = len(qa_stats)
    return qa_len


def get_qa_doc_id_from_question_id(question_id: int) -> int:
    """Get `doc_id` form QA stats for given `question_id`

    Args:
        question_id:

    Returns:
        `doc_id` form QA stats for given `question_id`

    """
    QaStat = Query()
    doc_id = tb_qa_stats.search(QaStat.question_id == question_id)[0].doc_id
    return doc_id


def save_user_answer(question_id: int, answer: str) -> None:
    """Update QA stats with user given answer for the `question_id`

    Args:
        question_id: Question ID
        answer: Answer

    Returns:
        None

    """
    doc_id = get_qa_doc_id_from_question_id(question_id=question_id)
    tb_qa_stats.update(dict(answer_user=answer), doc_ids=[doc_id])
    logger.info(f'Updated {question_id= } with {answer=}')


def get_all_qa() -> list[Document]:
    """Get all QA stats available

    Returns:
        List of all QA stats available

    """
    all_qa_docs = tb_qa_stats.all()
    return all_qa_docs


def get_audit_record(event: str = 'form_process', form_id: int = None, task: str = None,
                     sub_task: str = None) -> Document | None:
    """Get latest audit record that matches the given arguments

    Args:
        event: Event
        form_id: Form ID
        task: Task
        sub_task: Sub task

    Returns:
        Audit record or None
    """
    Audit = Query()
    audit_recs = tb_audit.search(
        (Audit.event == event) & (Audit.form_id == form_id) & (Audit.task == task) & (Audit.sub_task == sub_task))
    return audit_recs[-1] if audit_recs else None


def upsert_audit_record(delta: dict, doc_id: int = None) -> None:
    """Upsert audit record for given `doc_id`

    Args:
        delta: Delta to be applied
        doc_id: Document ID

    Returns:
        None

    """
    # logger.info(f'upsert_audit_record with {doc_id=}, {delta=}')
    if doc_id:
        tb_audit.update(delta, doc_ids=[doc_id])
    else:
        tb_audit.insert(delta)


def get_all_audit(form_id: int = None) -> list[Document]:
    """Get all audit records or all available for given form_id

    Args:
        form_id: Form ID

    Returns:
        All audit records or all available for given form_id

    """
    if form_id:
        Audit = Query()
        audit_records = tb_audit.search(Audit.form_id == form_id)
    else:
        audit_records = tb_audit.all()
    return sorted(audit_records, key=lambda x: x['start_time'], reverse=True)
