import dash
from dash import html, dcc, callback, Input, State, Output, ALL
import dash_bootstrap_components as dbc
import keras

dash.register_page(__name__, path='/train')

available_layers = [
    'Conv2D',
    'BatchNormalization',
    'Dense',
    'MaxPool2D',
    'Flatten',
    'Dropout',
]
available_optimizers = [
    optimizer for optimizer in dir(keras.optimizers) if not optimizer.startswith('_')
]
available_losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']


def create_class_upload(id_suffix):
    return dbc.Row(
        [
            dbc.Col(
                dcc.Upload(
                    id={'type': 'class-upload', 'index': id_suffix},
                    children=html.Div(['Перетащите или ', html.A('выберите файл')]),
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
                width=9,
            ),
            dbc.Col(
                dbc.Button(
                    "Удалить",
                    id={'type': 'delete-class', 'index': id_suffix},
                    color="danger",
                ),
                width=2,
            ),
        ],
        align="center",
        justify="between",
        className="g-0",
        style={'marginBottom': '10px'},
    )


def create_layer_input(id_suffix):
    return dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id={'type': 'layer-type', 'index': id_suffix},
                    options=[
                        {'label': layer, 'value': layer} for layer in available_layers
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
                html.Label(html.Strong('Классы')),
                html.Div(id='classes-wrapper', children=[create_class_upload(0)]),
                dbc.Button(
                    '+ Добавить класс',
                    color='dark',
                    id='add-class-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
                html.Label(id='test-split-label'),
                dcc.Input(
                    id='test-split-input',
                    max=100,
                    min=1,
                    type='range',
                    value=25,
                    step=1,
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
                                    for opt in available_optimizers
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
    ],
    className='center',
    style={'padding': '10px'},
)


@callback(
    Output("classes-wrapper", "children"),
    Input("add-class-button", "n_clicks"),
    State("classes-wrapper", "children"),
)
def add_class(n_clicks, children):
    if n_clicks >= len(children):
        new_child = create_class_upload(len(children))
        children.append(new_child)
    return children


@callback(
    Output("classes-wrapper", "children", allow_duplicate=True),
    Input({'type': 'delete-class', 'index': ALL}, 'n_clicks'),
    [
        State({'type': 'delete-class', 'index': ALL}, 'id'),
        State("classes-wrapper", "children"),
        State("add-class-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def delete_class(n_clicks, ids, children, add_btn_clicks):
    if n_clicks:
        index_to_delete = [i for i, id_dict in enumerate(ids) if n_clicks[i]]
        children = [
            child for i, child in enumerate(children) if i not in index_to_delete
        ]
        add_btn_clicks -= 1
    return children


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
    return html.Strong(f'Размер тестовой выборки: {val}%')


@callback(
    Output(component_id='epoch-count-label', component_property='children'),
    Input('epoch-count-input', 'value'),
)
def show_epoch_count(val):
    return html.Strong(f'Количество эпох: {val}')


def build_model():
    pass


@callback(
    [Output(component_id='train-form-error', component_property='children')],
    Input('train-button', 'n_clicks'),
    [
        State({'type': 'class-upload', 'index': ALL}, 'contents'),
        State('test-split-input', 'value'),
        State({'type': 'layer-type', 'index': ALL}, 'value'),
        State({'type': 'layer-params', 'index': ALL}, 'value'),
        State('optimizer', 'value'),
        State('learning-rate', 'value'),
        State('loss-function', 'value'),
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
    model_filename,
):
    if n_clicks:
        print("Class contents:", class_contents)
        print("Test sample size:", test_sample_size)
        print("Layer types:", layer_types)
        print("Layer params:", layer_params)
        print("Optimizer:", optimizer)
        print("Learning rate:", learning_rate)
        print("Loss function:", loss_function)
        if (
            class_contents[0] is None
            or layer_types[0] is None
            or layer_params[0] is None
            or not any([optimizer, loss_function, model_filename])
        ):
            return ('Ошибка, заполните форму правильно',)
    return ('',)
