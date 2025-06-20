import dash
from dash import Output, html, dcc, callback, Input, State
import dash_bootstrap_components as dbc
import keras
import torch
import prepare_image
from PIL import Image
import io
import base64
import numpy as np
import os

dash.register_page(__name__, path='/')
models_path = '/temp_models'

layout = html.Div(
    [
        html.H1('Тестирование нейросети'),
        dbc.Form(
            [
                html.P(id='error', style={'color': 'red'}),
                dcc.Upload(
                    id='upload-model',
                    children=dbc.Spinner(
                        html.Div(
                            id='model-upload-text',
                        )
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
                    children=dbc.Spinner(
                        html.Div(
                            id='image-upload-text',
                        )
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
                dbc.Button(
                    'Предсказать',
                    id='predict-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
            ],
            className='center test_form',
        ),
        dbc.Spinner(
            html.Div(
                id='prediction-result',
                className='center',
                style={'marginTop': '20px', 'minHeight': '60px', 'minWidth': '60px'},
            )
        ),
    ],
    className='center',
)


def decode_image(image_contents):
    content_string = image_contents.split(',')[1]
    decoded = base64.b64decode(content_string)
    img = np.array(Image.open(io.BytesIO(decoded)))[:, :, :3]
    return img


@callback(
    Output(component_id='model-upload-text', component_property='children'),
    Input('upload-model', 'contents'),
    State('upload-model', 'filename'),
)
def change_model_upload_text(_, filename):
    return filename if filename else ['Перетащите или ', html.A('выберите файл модели')]


@callback(
    Output(component_id='image-upload-text', component_property='children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
)
def change_image_upload_text(_, filename):
    return filename if filename else ['Перетащите или ', html.A('выберите фото')]


@callback(
    Output('predict-button', 'disabled'),
    Input('predict-button', 'n_clicks'),
)
def disable_button(n_clicks):
    return True if n_clicks > 0 else False


def save_model(model_contents, path):
    decoded = base64.b64decode(model_contents)
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    with open(path, 'wb') as file:
        file.write(decoded)


def predict_keras(path, img_array):
    prepared_img = prepare_image.prepare(img_array)
    model: keras.models.Sequential = keras.models.load_model(path)
    output = model.predict(np.array([prepared_img]), verbose=0)
    label, prob = np.argmax(output), output.max()
    label = prepare_image.tf_idx_label_map[label]
    return label, prob


def predict_torch(path, img_array):
    model = torch.jit.load(path)
    pipeline = prepare_image.PTHPipeLine(model)
    label, prob = pipeline(image=img_array)


@callback(
    [
        Output(component_id='error', component_property='children'),
        Output(component_id='prediction-result', component_property='children'),
        Output(
            component_id='predict-button',
            component_property='disabled',
            allow_duplicate=True,
        ),
        Output(
            component_id='predict-button',
            component_property='n_clicks',
            allow_duplicate=True,
        ),
    ],
    Input('predict-button', 'n_clicks'),
    [
        State('upload-model', 'contents'),
        State('upload-model', 'filename'),
        State('upload-image', 'contents'),
    ],
    prevent_initial_call=True,
)
def predict(n_clicks, model_contents, model_filename, image_contents):
    if n_clicks > 0 and not (model_contents and image_contents):
        return 'Ошибка, заполните форму правильно', ''

    if n_clicks > 0:
        path = f'{models_path}/{model_filename}'
        save_model(model_contents.split(',')[1], path)

        img_array = decode_image(image_contents)
        if model_filename.split('.')[1] == 'pth':
            label, prob = predict_torch(path, img_array)
        else:
            label, prob = predict_keras(path, img_array)
        os.remove(path)
        return (
            '',
            [
                html.Img(src=image_contents, style={'width': '300px'}),
                html.H3(f'Предсказанный класс: {label}'),
                html.P(f'Вероятность: {prob:.2f}'),
            ],
            False,
            0,
        )
    return '', ''
