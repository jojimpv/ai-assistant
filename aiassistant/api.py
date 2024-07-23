"""
Module to host FastAPI routes
"""
from nicegui import app

from aiassistant.core import parse_with_llm, qa_with_llm
from aiassistant.log import get_logger

logger = get_logger(__name__)


@app.get('/api/parse_form/{form_id}')
def parse_form(form_id: int):
    response = parse_with_llm(form_id=form_id)
    return dict(form_id=form_id, form_fields=response)


@app.get('/api/qa_form/{form_id}')
def qa_form(form_id: int):
    response = qa_with_llm(form_id=form_id)
    return dict(form_id=form_id, qa_stats=response)
