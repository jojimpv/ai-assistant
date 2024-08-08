import asyncio
import asyncio
import math

from dynaconf import settings
from nicegui import ui, events, app

import aiassistant.api as api
import aiassistant.audit as audit
from aiassistant.backend import process_uploads, process_parse_form, process_qa_form
from aiassistant.core import update_embedding
from aiassistant.database import tb_upload_stats, get_upload_stats, get_parse_stats, get_qa_stats, \
    get_next_question_answer, get_max_question_answer, save_user_answer, get_combined_qa, get_all_audit
from aiassistant.log import get_logger

logger = get_logger(__name__)
media_dir = settings.UPLOADS_DIR


class UiApp:
    def __init__(self):
        self.add_historic_forms()
        self.current_form = None
        self.current_form_max_question_index = 0
        self.current_question_index = 0
        self.current_question = ""
        self.current_question_id = ""
        self.current_answer = ""
        self.current_answer_auto = ""
        self.current_answer_user = ""
        self.form_preview_data = None
        self.audit_records = []
        self.audit_records_for_form = []

    def reset_current_form_info(self):
        self.current_form = None
        self.current_form_max_question_index = 0
        self.current_question_index = 0
        self.current_question = ""
        self.current_question_id = ""
        self.current_answer = ""
        self.current_answer_auto = ""
        self.current_answer_user = ""

    @property
    def user_input_previous_btn_visible(self):
        return self.current_question_index >= 1

    @property
    def user_input_previous_btn_not_visible(self):
        return self.current_question_index < 1

    @property
    def user_input_next_btn_visible(self):
        return self.current_question_index < self.current_form_max_question_index - 1

    @property
    def user_input_next_btn_not_visible(self):
        return self.current_question_index >= self.current_form_max_question_index - 1

    async def handle_uploads(self, e: events.UploadEventArguments):
        file_id = process_uploads(name=e.name, file_content=e.content.read())
        await asyncio.sleep(0.5)
        self.add_historic_forms()
        upload_dialog.close()
        ui.notify(f'Uploaded {e.name}')
        await self.handle_parse_form(form_id=file_id)
        await self.handle_qa_form(form_id=file_id)

    async def handle_parse_form(self, form_id: int):
        ui.notify(f'Starting form parsing for form_id: {form_id}')
        await process_parse_form(form_id=form_id)
        self.add_historic_forms()
        ui.notify(f'Completed form parsing for form_id: {form_id}.')

    async def handle_qa_form(self, form_id: int):
        ui.notify(f'Starting form QA for form_id: {form_id}')
        await process_qa_form(form_id=form_id)
        ui.notify(f'Completed form QA for form_id: {form_id}.')
        self.add_historic_forms()

    def handle_user_input(self, form_id):
        self.current_form = form_id
        self.current_form_max_question_index = get_max_question_answer(form_id=form_id)
        self.update_question_index()
        user_input_dialog.open()

    def handle_preview(self, form_id):
        self.form_preview_data = get_combined_qa(form_id=form_id)
        upload_stat = get_upload_stats(form_id=form_id)
        title = f'{form_id} | {upload_stat.get("name")}'
        preview_dialog.clear()
        with (preview_dialog, ui.card()):
            with ui.row().classes('w-full'):
                ui.label(title).classes('text-green text-lg')
                ui.space()
                with ui.button(on_click=preview_dialog.close, icon='close').props('flat color=blue'):
                    ui.tooltip('Close')
            for qa in ui_app.form_preview_data:
                question = qa['question']
                answer = qa['answer']
                source = qa['source']
                icon = 'auto_fix_normal' if source == 'auto' else 'person'
                with ui.expansion(text=question, icon=icon).props(
                        'header-class=text-blue dense switch-toggle-side').classes('w-full'):
                    ui.textarea(value=answer).props('readonly outlined input-class=h-0 hide-bottom-space').classes(
                        'w-full')
        preview_dialog.open()

    def update_question_index(self, by=0):
        new_question_index = self.current_question_index + by if by else 0
        # if new_question_index
        self.current_question_index = new_question_index
        qa = get_next_question_answer(form_id=self.current_form, index=new_question_index)
        self.current_question = qa['question']
        self.current_question_id = qa['question_id']
        self.current_answer_auto = qa['answer_auto']
        self.current_answer_user = qa.get('answer_user', '')

    def copy_answer_for_edit(self):
        self.current_answer_user = self.current_answer_auto

    def update_user_answer(self, answer: str):
        if self.current_answer_user.strip() != '' and self.current_answer_user.strip() != self.current_answer_auto.strip():
            save_user_answer(question_id=self.current_question_id, answer=answer)
            update_embedding(form_id=self.current_form, question_id=self.current_question_id, answer=answer)
            ui.notify('Corrections saved')

    def add_historic_forms(self):
        # logger.info(f'In add_historic_forms')
        forms_container.clear()
        for upload_stat in sorted(tb_upload_stats.all(), key=lambda rec: rec.doc_id, reverse=True):
            self.add_form_to_container(upload_stat.doc_id)

    def add_form_to_container(self, form_id):
        # logger.info(f'In add_form_to_container with form_id: {form_id}')
        upload_stat = get_upload_stats(form_id=form_id)
        form_status = upload_stat.get('status')
        title = f'{form_id} | {upload_stat.get("name")}'
        # Prepare parse tab data
        parse_stats = get_parse_stats(form_id=form_id)
        parse_stat_columns = [
            dict(name='field_name', label='Field Names', field='field_name', required=True, align='left'),
        ]
        parse_stat_rows = [
            dict(
                field_name=x.get('field')
            ) for x in parse_stats
        ]
        # Prepare qa tab data
        qa_stats = get_qa_stats(form_id=form_id)
        qa_stat_columns = [
            dict(name='question', label='Question', field='question', required=True, align='left'),
            dict(name='answer_auto', label='Answer', field='answer_auto', required=True, align='left'),
        ]
        qa_stat_columns_user = [
            dict(name='question', label='Question', field='question', required=True, align='left'),
            dict(name='answer_user', label='Answer', field='answer_user', required=False, align='left'),
        ]
        qa_stat_rows = [
            dict(
                doc_id=x.doc_id,
                form_id=x.get('form_id'),
                question_id=x.get('question_id'),
                question=get_parse_stats(form_id, parse_id=x.get('question_id'))['field'],
                answer_auto=x.get('answer_auto'),
                answer_user=x.get('answer_user', ""),
            ) for x in qa_stats
        ]

        # Build forms container UI
        with forms_container:
            with ui.expansion(str(title), icon='feed').props('header-class=text-green').classes('w-full bg-green-50'):
                upload_stat_columns = [
                    dict(name='name', label='Form Name', field='name', required=True, align='left'),
                    dict(name='file_name', label='File Name', field='file_name', required=True, align='left'),
                    dict(name='file_size', label='File Size (KB)', field='file_size', required=True, align='center'),
                    dict(name='status', label='Status', field='status', required=True, align='left'),
                ]
                upload_stat_rows = [
                    dict(
                        name=upload_stat.get('name'),
                        file_name=upload_stat.get('file_name'),
                        file_size=int(math.ceil(upload_stat.get('file_size') / 1024)),
                        status=form_status,
                    )
                ]
                with ui.column():
                    # Upload stats table
                    ui.table(columns=upload_stat_columns, rows=upload_stat_rows, row_key='name')
                    # Next action button creation
                    with ui.row():
                        if form_status == settings.FORM_STATUS_QA:
                            with ui.column():
                                ui.button('User Input (Review)',
                                          on_click=lambda: self.handle_user_input(form_id=form_id),
                                          icon='play_arrow').props('flat color=blue')
                                ui.button('Preview Answers', on_click=lambda: self.handle_preview(form_id=form_id),
                                          icon='tour').props('flat color=blue')
                        ui.button('Track form process',
                                  on_click=lambda: self.load_audit_records(form_id=form_id),
                                  icon='play_arrow').props('flat color=blue')
                # Tabs for stage status details
                with ui.tabs().props('align=left').classes('w-full bg-green-500 text-white') as tabs:
                    parse_tab = ui.tab('Parse')
                    qa_tab = ui.tab('QA (Auto)')
                    qa_tab_user = ui.tab('QA (User)')
                with ui.tab_panels(tabs, value=parse_tab).classes('w-full bg-green-50'):
                    with ui.tab_panel(parse_tab):
                        ui.table(columns=parse_stat_columns, rows=parse_stat_rows, row_key='field_name')
                    with ui.tab_panel(qa_tab):
                        ui.table(columns=qa_stat_columns, rows=qa_stat_rows, row_key='field_name')
                    with ui.tab_panel(qa_tab_user):
                        ui.table(columns=qa_stat_columns_user, rows=qa_stat_rows, row_key='field_name')

    def load_audit_records(self, form_id):
        self.audit_records_for_form = [audit.get_formatted_audit_row(x) for x in get_all_audit(form_id=form_id)]
        audit_dialog.clear()
        with (audit_dialog, ui.card().style('width: 1200px; min-width: fit-content;')):
            ui.table(
                columns=audit.audit_columns_form,
                rows=self.audit_records_for_form,
                row_key='event'
            ).props('dense table-header-class="text-blue" title-class="text-green"')
        audit_dialog.open()

    def filter_audit_records(self):
        self.audit_records = [x for x in self.audit_records if x['event'] == 'form_process']


# UI
forms_container = ui.row().classes('w-full')
# user_input_container = ui.row().classes('w-full')
with forms_container:
    ui.label('Place holder for forms')

# Initiate UiApp to load historic forms to above container
ui_app = UiApp()

# New file upload dialog (hidden by default)
upload_dialog = ui.dialog()
with (upload_dialog, ui.card()):
    ui.upload(on_upload=ui_app.handle_uploads, auto_upload=True, label='Select file'
              ).props('accept=.pdf').classes('max-w-full')
    # ui.button('Done', on_click=upload_dialog.close)

# User input dialog (hidden by default)
user_input_dialog = ui.dialog()
with (user_input_dialog, ui.card().style('width: 1200px; max-width: none')):
    with ui.row().classes('w-full'):
        ui.button('Previous', on_click=lambda: ui_app.update_question_index(by=-1), icon='fast_rewind'
                  ).props('flat color=blue').bind_visibility_from(ui_app, 'user_input_previous_btn_visible')
        ui.button('Previous', icon='fast_rewind').props('disable flat color=blue').bind_visibility_from(ui_app,
                                                                                                        'user_input_previous_btn_not_visible')
        ui.button('Next', on_click=lambda: ui_app.update_question_index(by=1), icon='fast_forward'
                  ).props('flat color=blue').bind_visibility_from(ui_app, 'user_input_next_btn_visible')
        ui.button('Next', icon='fast_forward').props('disable flat color=blue').bind_visibility_from(ui_app,
                                                                                                     'user_input_next_btn_not_visible')
        ui.space()
        ui.button(on_click=lambda: (ui_app.reset_current_form_info(), user_input_dialog.close()), icon='close').props(
            'flat color=blue')
    ui.separator()
    ui.markdown().bind_content_from(ui_app, 'current_question', backward=lambda q: f'**Question:** {q} ?')
    ui.separator()
    ui.markdown('**Answer:**')
    with ui.splitter(limits=(10, 500)).classes('w-full') as splitter:
        with splitter.before:
            ui.textarea(placeholder='AI generated Answer'
                        ).bind_value_from(ui_app, 'current_answer_auto'
                                          ).props('readonly'
                                                  ).props('label="AI generated"').style('width: 98%')
        with splitter.after:
            user_correction = ui.textarea(placeholder='Edit to replace Answer'
                                          ).bind_value(ui_app, 'current_answer_user'
                                                       ).classes('ml-2').props('label="User corrections (editable)"'
                                                                               ).style('width: 98%').props('clearable')
    with ui.row():
        with ui.button('Copy to edit', on_click=ui_app.copy_answer_for_edit).props('flat color=blue'):
            ui.tooltip('Copy text from "AI generated" panel to "User corrections" panel')
        ui.space()
        with ui.button('Save', on_click=lambda e: ui_app.update_user_answer(user_correction.value)
                       ).props('flat color=blue'):
            ui.tooltip('Save edits in "User corrections" panel')

# Preview dialog (hidden by default)
preview_dialog = ui.dialog()  # .props('maximized')

# Audit for a single form (hidden by default)
audit_dialog = ui.dialog().classes('w-full')

# UI Header
main_header = ui.header(elevated=True).style('background-color: green').classes('items-center justify-between')
with main_header:
    with ui.button('New form', on_click=upload_dialog.open, icon='add').props('flat color=white'):
        ui.tooltip('Upload new form')
    ui.space()
    ui.label('AI Assistant').classes('text-lg font-bold')
    ui.space()
    with ui.link(target=audit.load_audit_page, new_tab=True):
        with ui.button(icon='checklist_rtl').props('flat color=white'):
            ui.tooltip('Audit data process timings')

    with ui.button(icon='power_settings_new', on_click=lambda: (ui.notify('App Shutdown!'), app.shutdown())).props(
            'flat color=white'):
        ui.tooltip('Shutdown the app!')

logger.info(f'API module loaded from {api}')
logger.info(f'Audit module loaded from {audit}')

ui.run(reload=False, show=False, title='AI Assistant', favicon='📝')
