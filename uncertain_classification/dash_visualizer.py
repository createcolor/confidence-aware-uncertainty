from typing import Dict, List
import argparse
from pathlib import Path
import json
import plotly.graph_objs as go
import dash
from dash.dependencies import Input, Output
from dash import dcc, html
import numpy as np

from uncertain_classification.rejection_curves import (get_accs_throwaway_rates,  # pylint: disable=import-error
                                                       calculate_aucs)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

path2rejection_curves_data = ("./uncertain_classification/rejection_curves_args.json")
rej_curves_data_dir = Path("./results/rejection_curves")

with open(path2rejection_curves_data, "r", encoding="utf-8") as f:
    rej_curves_paths = json.load(f)

paths = {method: rej_curves_data_dir / data["path2save_reg_curves_info"]
         for method, data in rej_curves_paths["UE_methods"].items()}
colorpalette = ["#026374", "#099396", "#94D2BD", "#EE9B00", "#CA6702", "#BB3F03"]


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig


app.layout = html.Div([
    html.H1("Welcome to the neural network feature space analyzer!",
            style={'font-size': 20, 'width': '100%', 'margin': '10pt',
                   'textAlign': 'center', 'font': "Courier New"}),
    html.Plaintext('Here you can view heatmaps and rejection curves.',
                   style={'font-size': 15, 'width': '100%', 'textAlign': 'center',
                          'font_family': "Courier New"}),

    ########
    html.Div(children=[
        html.Label('Rejection curves types:'),
        dcc.Checklist(id='rejection_curve_mode', options=list(paths.keys()),
                      labelStyle={'font-size': 13},),
        ],
        style={'width': '15%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(id='rejection_curve', figure=go.Figure()),
             style={'width': '74%', 'display': 'inline-block'}),
    html.Div(id='auc'),
    html.Br(),
    ########
    html.Div(children=[
            html.Label('Net id:'),
            dcc.RadioItems(
                id='net_id',
                options=[{"label": i, "value": i} for i in range(50)],
                value=0,
                labelStyle={'font-size': 13, 'margin': '5px', 'display': 'inline-block'},
                persistence=True
            ),],
            style={'width': '100%', 'display': 'inline-block'}),
    html.Br(),

    html.Div(children=[
        html.Label('Reagents:'),
        dcc.Checklist(
            id='reagent',
            options=["all", "A", "B", "D", "O(I)", "A(II)", "B(III)",
                     "C", "c", "E", "e", "Cw", "K", "k"],
            value=["all"],
            labelStyle={'font-size': 13},
        ),
        ],
        style={'width': '10%', 'display': 'inline-block'}),
    html.Br(),
    ########
    dcc.Store(id='misclas_alvs'),
    dcc.Store(id='net_outputs'),
    dcc.Store(id='threshold'),
    dcc.Store(id='features'),
    dcc.Store(id='alvs_info'),
    dcc.Store(id='pca_dim'),
    dcc.Store(id='test_markup'),
])


@app.callback([Output('misclas_alvs', 'data'), Output('net_outputs', 'data'),
               Output('threshold', 'data'), Output('features', 'data'),
               Output('alvs_info', 'data')], [Input('net_id', 'value'),])
def update_net_info(net_id, path2features, path2alvs_info, path2nets_outputs,
                    path2misclas_alvs, path2thresholds):
    nets_data = []
    for path in (path2features, path2alvs_info, path2nets_outputs,
                 path2misclas_alvs, path2thresholds):
        with open(path, 'r', encoding="utf-8") as nets_file:
            nets_data.append(json.load(nets_file)[net_id])
    return nets_data


@app.callback(Output('rejection_curve', 'figure'), Output('auc', 'children'),
              [Input('rejection_curve_mode', 'value'), Input('misclas_alvs', 'data')])
def update_graph(rejection_curve_mode, misclas_alvs):
    uncertainty_types = {}
    for mode in rejection_curve_mode:
        with open(paths[mode], "r", encoding="utf-8") as uncertainty_file:
            uncertainty_types[mode] = json.load(uncertainty_file)

    traces, layout, aucs = plot_rejection_curves(uncertainty_types, misclas_alvs,
                                                 display_stds=False)

    # fig = go.Figure({'data': traces, 'layout': layout})
    # path2save_plot = "results/rej_curves.pdf"

    message_about_aucs = "The areas above the rejection curves (x 10^4) are: \n"
    for label, auc in aucs.items():
        message_about_aucs += f"{label}: {round(1.0 - auc, 8) * 10 ** 4}; _ "
    return {'data': traces, 'layout': layout}, message_about_aucs


def plot_rejection_curves(data_dict: Dict, misclas_alvs, display_stds=False,
                          curves_are_averaged=True):
    '''
    Plots accuracy-rejection curves (dependence of accuracies on the left test set part
    after the most uncertain alveoluses are thrown away).

    TODO: take the reagent into consideration
    TODO: not averaged rejection curves data
    TODO: display std button
    '''
    def choose_color(used_colors_ids: List):
        if len(used_colors_ids) == len(colorpalette):
            used_colors_ids = []
        color_id = np.random.choice(list(set(range(len(colorpalette))) - set(used_colors_ids)))
        used_colors_ids.append(color_id)

        color_hex = colorpalette[color_id]
        color_rgb_values = [int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
        return color_rgb_values, used_colors_ids

    traces = []
    accuracies_dict, throwaway_rates_dict = {}, {}
    used_colors_ids = []

    layout = go.Layout(template=None, title=dict(text="Rejection curves"),
                       xaxis=dict(title='Throwaway rate'), yaxis=dict(title='Accuracy'))

    for label, data in data_dict.items():
        color_rgb, used_colors_ids = choose_color(used_colors_ids)

        if curves_are_averaged:
            throwaway_rates = sorted(list(map(float, list(data.keys()))))
            means, stds = [], []
            for throwaway_rate in throwaway_rates:
                means.append(data[str(throwaway_rate)][0])
                stds.append(data[str(throwaway_rate)][1])

            if display_stds:
                std_upper = [means[i] + std for i, std in enumerate(stds)]
                std_lower = [means[i] - std for i, std in enumerate(stds)]

                traces.append(go.Scatter(
                    x=throwaway_rates + throwaway_rates[::-1],  # x, then x reversed
                    y=std_upper + std_lower[::-1],  # upper, then lower reversed,
                    fill='tozerox', line=dict(width=0), showlegend=False,
                    fillcolor=f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.4)",))

            traces.append(go.Scatter(x=throwaway_rates, y=means, name=label,
                                     line=dict(color=(f"rgba({color_rgb[0]},{color_rgb[1]},"
                                                      f"{color_rgb[2]},1)"),
                                               dash="solid", width=1.7)))

        else:
            accuracies, throwaway_rates = get_accs_throwaway_rates(data, test_markup, misclas_alvs)
            traces.append(go.Scatter(x=throwaway_rates, y=accuracies, name=label,
                          line=dict(color=f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)",
                                    dash="solid", width=1.7)))

        accuracies_dict[label] = means
        throwaway_rates_dict[label] = throwaway_rates

    aucs = calculate_aucs(accuracies_dict, throwaway_rates_dict, min_accuracy=0.0)
    return traces, layout, aucs


def parse_args():
    parser = argparse.ArgumentParser("Usage example: python nn/dash_visualize_features.py "
                                     "-ctrain nn/net_configs/MobileNet_10_10_80/MobileNet_"
                                     "900ep_try5.json -ndp /mnt/nuberg/datasets/medical_data/nets/"
                                     "MobileNet_10_10_80_try5 -tm markup/test_dataset_10_10_80.json"
                                     " -n MobileNet_10_10_80_try5_600ep_4")

    parser.add_argument('-r', '--dataset_dir',
                        default=Path("/mnt/nuberg/datasets/medical_data/alvs_dataset_all"),
                        type=Path, help='Path to images')
    parser.add_argument('-tm', '--test_markup', default=Path("markup/BloodyWell/"
                                                             "test_dataset_10_10_80.json"),
                        type=Path, help='Path to test dataset markup')

    parsed_args = parser.parse_args()
    return parsed_args


args = parse_args()

with open(args.test_markup, "r", encoding="utf-8") as f:
    test_markup = json.load(f)

if __name__ == '__main__':
    app.run_server(debug=True)
