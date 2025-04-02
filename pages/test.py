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
                        'width': '50%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                    },
                    multiple=False,
                ),
                dcc.Upload(
                    id='upload-image',
                    children=html.Div(['Перетащите или ', html.A('выберите фото')]),
                    style={
                        'width': '50%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px',
                    },
                    multiple=False,
                ),
                html.Div(id='output-image-upload'),
                html.Button('Предсказать', id='predict-button', n_clicks=0),
            ],
            className='center',
        ),
        html.Div(id='prediction-result'),
    ],
    className='center',
)


# @callback(Output('analytics-output', 'children'), Input('analytics-input', 'value'))
# def update_city_selected(input_value):
#     return f'You selected: {input_value}'
