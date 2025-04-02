import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/')

layout = html.Div(
    [
        html.H1('Тестирование нейросети'),
        dbc.Form(
            [
                dcc.Upload(
                    id='upload-image',
                    children=html.Div(
                        ['Перетащите или ', html.A('выберите файл модели')]
                    ),
                    style={
                        'height': '60px',
                        'width': '100%',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'padding': '0 10px',
                    },
                    multiple=False,
                ),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div(['Перетащите или ', html.A('выберите фото')]),
                    style={
                        'height': '60px',
                        'width': '100%',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'padding': '0 10px',
                    },
                    multiple=False,
                ),
                dbc.Button(
                    'Предсказать',
                    id='predict-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
            ],
            className='center test_form',
        ),
        html.Div(id='prediction-result'),
    ],
    className='center',
)


# @callback(Output('analytics-output', 'children'), Input('analytics-input', 'value'))
# def update_city_selected(input_value):
#     return f'You selected: {input_value}'
