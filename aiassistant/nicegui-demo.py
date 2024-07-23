from nicegui import ui

preview_dialog = ui.dialog()
title = '20240717161633 | sample-form-1.pdf'
form_preview_data = [
    dict(question='Q1', answer='A1', source='auto'),
    dict(question='Q2', answer='A2', source='user'),
]

with (preview_dialog, ui.card()):
    with ui.row().classes('w-full'):
        ui.label(title).classes('text-green text-lg')
        ui.space()
        ui.button('close', on_click=preview_dialog.close, icon='close')
    for qa in form_preview_data:
        question = qa['question']
        answer = qa['answer']
        source = qa['source']
        icon = 'auto_fix_normal' if source == 'auto' else 'person'
        with ui.expansion(text=question, icon=icon).props('header-class=text-blue dense switch-toggle-side header-style="background: green"').classes('w-full'):
            ui.textarea(value=answer).props('readonly').classes('w-full').props('outlined input-class=h-0 hide-bottom-space dense')
    ui.icon("person")
ui.button('Preview', on_click=preview_dialog.open)

ui.run(port=9000, reload=False)
