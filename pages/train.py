import base64
import dash
from dash import html, dcc, callback, Input, State, Output, ALL
import dash_bootstrap_components as dbc
import keras
import zipfile
import os
from PIL import Image
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from prepare_image import prepare, tf_label_idx_map
import plotly.graph_objs as go
import threading

dash.register_page(__name__, path='/train')

available_layers = {
    'Conv2D': keras.layers.Conv2D,
    'BatchNormalization': keras.layers.BatchNormalization,
    'Dense': keras.layers.Dense,
    'MaxPool2D': keras.layers.MaxPool2D,
    'Flatten': keras.layers.Flatten,
    'Dropout': keras.layers.Dropout,
}
available_optimizers = {
    'Adam': keras.optimizers.Adam,
    'Nadam': keras.optimizers.Nadam,
    'RMSprop': keras.optimizers.RMSprop,
    'SGD': keras.optimizers.SGD,
}
available_losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
status_data = {'status': '', 'train_history': {}}


def create_layer_input(id_suffix):
    return dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id={'type': 'layer-type', 'index': id_suffix},
                    options=[
                        {'label': layer, 'value': layer}
                        for layer in available_layers.keys()
                    ],
                    placeholder='Выберите тип слоя',
                )
            ),
            dbc.Col(
                dbc.Input(
                    id={'type': 'layer-params', 'index': id_suffix},
                    type='text',
                    placeholder='Параметры слоя (key=value)',
                ),
                width=5,
            ),
            dbc.Col(
                dbc.Button(
                    "Удалить",
                    id={'type': 'delete-layer', 'index': id_suffix},
                    color="danger",
                ),
                width=2,
            ),
        ],
        align="center",
        justify="between",
        style={'marginBottom': '10px'},
    )


layout = html.Div(
    [
        html.H1('Обучение нейросети'),
        dbc.Form(
            [
                html.P(id='train-form-error', style={'color': 'red'}),
                html.Label(html.Strong('Данные')),
                dcc.Upload(
                    id='data-upload',
                    children=dbc.Spinner(html.P(id='data-upload-text')),
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
                html.Label(id='test-split-label'),
                dcc.Input(
                    id='test-split-input',
                    max=1,
                    min=0.01,
                    type='range',
                    value=0.25,
                    step=0.01,
                ),
                html.Label(html.Strong('Слои')),
                html.Div(id='layers-wrapper', children=[create_layer_input(0)]),
                dbc.Button(
                    '+ Добавить слой',
                    color='dark',
                    id='add-layer-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
                html.Label(html.Strong('Функция потерь')),
                dcc.Dropdown(
                    id="loss-function",
                    options=[
                        {'label': loss, 'value': loss} for loss in available_losses
                    ],
                    placeholder='Выберите функцию потерь',
                ),
                html.Label(html.Strong('Оптимизатор и learning rate')),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="optimizer",
                                options=[
                                    {'label': opt, 'value': opt}
                                    for opt in available_optimizers.keys()
                                ],
                                placeholder='Выберите оптимизатор',
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dbc.Input(
                                id="learning-rate",
                                type="number",
                                placeholder="Введите learning rate",
                                max=1,
                                min=0.0001,
                                step=0.0001,
                                value=0.001,
                            ),
                            width=6,
                        ),
                    ],
                    style={'display': 'flex'},
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
                html.Label(
                    html.Strong(
                        'Название файла (без указания расширения, по умолчанию .keras)'
                    )
                ),
                dbc.Input(id='model-name-input', placeholder='Напишите название файла'),
                dbc.Button(
                    'Обучить',
                    id='train-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
            ],
            className='center test_form',
        ),
        dcc.Interval(
            id='interval-component',
            interval=1000,
            n_intervals=0,
        ),
        html.P(id='train-status'),
        dcc.Graph(id='loss-graph'),
        dcc.Graph(id='accuracy-graph'),
        dbc.Button("Скачать файл модели", id="download-model-btn"),
        dcc.Download(id='download-model'),
    ],
    className='center',
    style={'padding': '10px'},
)


@callback(
    Output("data-upload-text", "children"),
    Input("data-upload", "contents"),
    State("data-upload", "filename"),
)
def add_upload_text(contents, filename):
    return filename if filename else ['Перетащите или ', html.A('выберите архив')]


@callback(
    Output("layers-wrapper", "children"),
    Input("add-layer-button", "n_clicks"),
    State("layers-wrapper", "children"),
)
def add_layer(n_clicks, children):
    if n_clicks >= len(children):
        new_child = create_layer_input(len(children))
        children.append(new_child)
    return children


@callback(
    Output(component_id='test-split-label', component_property='children'),
    Input('test-split-input', 'value'),
)
def show_test_split_val(val):
    return html.Strong(f'Размер тестовой выборки: {val}')


@callback(
    Output(component_id='epoch-count-label', component_property='children'),
    Input('epoch-count-input', 'value'),
)
def show_epoch_count(val):
    return html.Strong(f'Количество эпох: {val}')


def load_data(data_zip) -> tuple[np.ndarray, np.ndarray]:
    path = './data'
    with open(f'{path}.zip', 'wb') as file:
        file.write(base64.b64decode(data_zip))
    with zipfile.ZipFile(f'{path}.zip', 'r') as zip_ref:
        zip_ref.extractall(path)
    train_images = []
    train_labels = []
    folders = [folder for folder in os.listdir(path) if folder[0] != '.']
    for i in range(len(folders)):
        folder = folders[i]
        for image_path in os.listdir(f'{path}/{folder}'):
            image = Image.open(f'{path}/{folder}/{image_path}')
            try:
                image = Image.fromarray(prepare(np.array(image)))
            except ValueError:
                continue
            train_images.append(np.array(image))
            train_labels.append(tf_label_idx_map[folder])
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    os.remove('./data.zip')
    shutil.rmtree('./data')
    return train_images, train_labels


def parse_params(params: str) -> dict:
    if params == '':
        return {}
    splitted_params = params.split(' ')
    params_dict = {}
    for param in splitted_params:
        key, val = param.split('=')
        if val.isdigit():
            val = int(val)
        elif '(' in val:
            val = tuple(map(int, val.replace('(', '').replace(')', '').split(',')))
        elif '[' in val:
            val = list(map(int, val.replace('[', '').replace(')', '').split(',')))
        elif '.' in val:
            val = float(val)
        params_dict[key] = val
    return params_dict


def build_model(input_shape, layers, params) -> keras.models.Sequential:
    model = keras.models.Sequential()
    model.add(keras.Input(shape=input_shape))
    for idx, layer in enumerate(layers):
        layer_params_dict = parse_params(params[idx])
        print(layer_params_dict)
        model.add(available_layers[layer](**layer_params_dict))
    return model


def train_model(
    model: keras.models.Sequential,
    class_contents,
    test_sample_size,
    loss,
    optimizer,
    learning_rate,
    epochs,
    model_filename,
    status_data,
):
    status_data['status'] = 'Загрузка и обработка данных'
    train_images, train_labels = load_data(class_contents)
    X_train, X_test, y_train, y_test = train_test_split(
        train_images, train_labels, test_size=test_sample_size, random_state=42
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'./{model_filename}.keras',
        save_weights_only=False,
        save_best_only=True,
        save_freq="epoch",
        verbose=1,
    )

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(logs.items())
            for key, value in logs.items():
                if key not in status_data['train_history']:
                    status_data['train_history'][key] = []
                status_data['train_history'][key].append(value)

    optimizer = available_optimizers[optimizer](learning_rate=learning_rate)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy'],
    )
    status_data['status'] = 'Обучение модели'
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        callbacks=[checkpoint_callback, CustomCallback()],
        validation_data=[X_test, y_test],
        batch_size=128,
        # class_weight=model3_class_weight,
        verbose=1,
    )
    status_data['status'] = 'Обучение завершено'


@callback(
    Output('train-button', 'disabled', allow_duplicate=True),
    Input('train-button', 'n_clicks'),
    prevent_initial_call=True,
)
def disable_train_button(n_clicks):
    return True if n_clicks else False


@callback(
    [
        Output(component_id='train-form-error', component_property='children'),
        Output('train-button', 'disabled'),
    ],
    Input('train-button', 'n_clicks'),
    [
        State('data-upload', 'contents'),
        State('test-split-input', 'value'),
        State({'type': 'layer-type', 'index': ALL}, 'value'),
        State({'type': 'layer-params', 'index': ALL}, 'value'),
        State('optimizer', 'value'),
        State('learning-rate', 'value'),
        State('loss-function', 'value'),
        State('epoch-count-input', 'value'),
        State('model-name-input', 'value'),
    ],
)
def handle_form(
    n_clicks,
    class_contents,
    test_sample_size,
    layer_types,
    layer_params,
    optimizer,
    learning_rate,
    loss_function,
    epochs,
    model_filename,
):
    if n_clicks:
        model = build_model(
            input_shape=(250, 250, 3),
            layers=['Conv2D', 'Flatten', 'Dense', 'Dense'],
            params=[
                'kernel_size=(4,4) filters=1',
                '',
                'units=150 activation=relu',
                'units=3 activation=softmax',
            ],
        )
        thread = threading.Thread(
            target=train_model,
            args=(
                model,
                class_contents,
                test_sample_size,
                loss_function,
                optimizer,
                learning_rate,
                epochs,
                'model',
                status_data,
            ),
        )
        thread.start()
        # thread.join()

        if (
            class_contents[0] is None
            or layer_types[0] is None
            or layer_params[0] is None
            or not any([optimizer, loss_function, model_filename])
        ):
            return ('Ошибка, заполните форму правильно', False)
        status_data['status'] = ''
        status_data['train_history'] = {}
    return ('', False)


@callback(
    Output('train-status', 'children'), Input('interval-component', 'n_intervals')
)
def update_status(n_intervals):
    return status_data['status']


@callback(
    [Output('loss-graph', 'figure'), Output('accuracy-graph', 'figure')],
    Input('interval-component', 'n_intervals'),
)
def update_graph(n_intervals):
    loss_traces = []
    accuracy_traces = []
    for metric, values in status_data['train_history'].items():
        if metric[-4:] == 'loss':
            loss_traces.append(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines+markers',
                    name=metric,
                )
            )
        else:
            accuracy_traces.append(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines+markers',
                    name=metric,
                )
            )

    loss_layout = go.Layout(
        title="История обучения",
        xaxis=dict(title="Эпоха"),
        yaxis=dict(title="Ошибка"),
    )
    accuracy_layout = go.Layout(
        title="История обучения",
        xaxis=dict(title="Эпоха"),
        yaxis=dict(title="Accuracy"),
    )
    return {'data': loss_traces, 'layout': loss_layout}, {
        'data': accuracy_traces,
        'layout': accuracy_layout,
    }


@callback(
    Output('download-model-btn', 'disabled'),
    Input('interval-component', 'n_intervals'),
)
def download_btn_disable(n_intervals):
    return status_data['status'] != 'Обучение завершено'


@callback(
    Output('download-model', 'data'),
    Input('download-model-btn', 'n_clicks'),
    State('model-name-input', 'value'),
)
def add_download_model_btn(n_clicks, filename):
    if n_clicks:
        return dcc.send_file(f'./{filename}.keras')
    return None
