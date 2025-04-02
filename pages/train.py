import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/train')

layout = html.Div(
    [
        html.H1('Обучение нейросети'),
    ],
    className='center',
)
