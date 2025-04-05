import dash
from dash import html, dcc, callback, Input, State, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/train')

layout = html.Div(
    [
        html.H1('Обучение нейросети'),
        dbc.Form(
            [
                html.P(id='error', style={'color': 'red'}),
                html.Div(id='classes-wrapper'),
                dbc.Button(
                    '+ Добавить класс',
                    color='dark',
                    id='add-class-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
                dbc.Button(
                    '+ Добавить слой',
                    color='dark',
                    id='add-layer-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
                html.Label(id='epoch-count-label'),
                dcc.Input(
                    id='epoch-count-input',
                    max=100,
                    min=1,
                    type='range',
                    value=10,
                    step=1,
                ),
                dbc.Button(
                    'Обучить',
                    id='train-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
            ],
            className='center test_form',
        ),
    ],
    className='center',
)


@callback(
    Output(component_id='epoch-count-label', component_property='children'),
    Input('epoch-count-input', 'value'),
)
def show_epoch_count(val):
    return f'Количество эпох: {val}'
