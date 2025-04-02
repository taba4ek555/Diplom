import dash
from dash import Output, html, dcc, callback, Input, State
import dash_bootstrap_components as dbc
import prepare_image
from PIL import Image
import io
import base64
import numpy as np

dash.register_page(__name__, path='/')

layout = html.Div(
    [
        html.H1('Тестирование нейросети'),
        dbc.Form(
            [
                html.P(id='error', style={'color': 'red'}),
                dcc.Upload(
                    id='upload-model',
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


def save_model_from_bin(filename, data, path):
    pass


def decode_image(image_contents):
    content_string = image_contents.split(',')[1]
    decoded = base64.b64decode(content_string)
    img = np.array(Image.open(io.BytesIO(decoded)))
    return img


@callback(
    [
        Output(component_id='error', component_property='children'),
        Output(component_id='prediction-result', component_property='children'),
    ],
    Input('predict-button', 'n_clicks'),
    [
        State('upload-model', 'contents'),
        State('upload-model', 'filename'),
        State('upload-image', 'contents'),
        State('upload-image', 'filename'),
    ],
)
def predict(n_clicks, model_contents, model_filename, image_contents, image_filename):
    if n_clicks > 0 and not (model_contents and image_contents):
        return 'Ошибка, заполните форму правильно', html.Div()
    if n_clicks > 0:
        save_model_from_bin(model_filename, model_contents, '/models')
        if model_filename.split('.')[1] == 'pth':
            pipeline = prepare_image.PTHPipeLine()
        else:
            pass
        img_array = prepare_image.prepare(decode_image(image_contents))
        print(img_array)
    return '', ''
    # prediction = model.predict(img_array)
    # predicted_class = np.argmax(prediction, axis=1)

    # return html.Div(
    #     [
    #         html.H3(f'Предсказанный класс: {predicted_label}'),
    #         html.P(f'Вероятность: {prediction[0][predicted_class[0]]:.2f}'),
    #     ]
    # )
