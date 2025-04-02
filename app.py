import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


pages_names = {'Test': 'Тестирование', 'Train': 'Обучение'}
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])


navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(pages_names[page["name"]], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        label="Страницы",
    ),
    brand="Face recognition",
    color="primary",
    dark=True,
    className="mb-2",
)

app.layout = html.Div(
    [
        navbar,
        dash.page_container,
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
