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
                dcc.Input(
                    id={'type': 'layer-params', 'index': id_suffix},
                    type='text',
                    placeholder='Параметры слоя',
                )
            ),
            dbc.Col(
                dbc.Button(
                    "Удалить",
                    id={'type': 'delete-layer', 'index': id_suffix},
                    color="danger",
                )
            ),
        ],
        align="center",
        justify="between",
        className="g-0",
        style={'marginBottom': '10px'},
    )


layout = html.Div(
    [
        html.H1('Обучение нейросети'),
        dbc.Form(
            [
                html.P(id='error', style={'color': 'red'}),
                html.Div(id='classes-wrapper', children=[create_class_upload(0)]),
                dbc.Button(
                    '+ Добавить класс',
                    color='dark',
                    id='add-class-button',
                    n_clicks=0,
                    style={'width': '100%'},
                ),
                html.Div(id='layers-wrapper', children=[create_layer_input(0)]),
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
    Output(component_id='epoch-count-label', component_property='children'),
    Input('epoch-count-input', 'value'),
)
def show_epoch_count(val):
    return f'Количество эпох: {val}'
