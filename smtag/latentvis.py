import json
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import torch
from torch.nn import functional as F
from transformers import (
    AutoTokenizer, AutoModel
)

from .models.vae import Twin

from .config import config

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class LatentVis:

    def __init__(self, model):
        self.model = model
        self.model.eval()
        logging.info(f"model config:\n{model.config}")
        self.L,  self.F, self.Z = model.config.seq_length, model.config.hidden_features, model.config.z_dim
        self.weights = self.get_weights()

    def get_weights(self):
        named_modules = dict(self.model.named_modules())
        fc = named_modules.get('encoder.fc_z_1')
        if fc is None:
            fc = named_modules['encoder.fc_z_mean']
        weights = fc.weight
        logging.info(f"weights with size {weights.size()}")
        return weights

    def contribution(self, inputs):
        # suppose Batch size = 1, squeezed out
        # last_hidden_state H is L x F
        # y is (L*F)
        # w is Z x (L*F)
        # z is Z
        # return r Z x L of contributions of each token position given latent var
        # given z_i i in Z what are the importance of each token t_k, k in L
        # z_i = sum over j in L*F of w_ij * y_j
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        twin_inputs = {
            'input_ids': [input_ids] * 2,
            'attention_mask': [attention_mask] * 2,
        }
        outputs = self.model(**twin_inputs)
        H = outputs.hidden_before_latent[0].squeeze(0)
        Z = outputs.representations[0].squeeze(0)
        # weights with size torch.Size([1024, 65536])
        # RuntimeError: shape '[1024, 20, 256]' is invalid for input of size 67108864
        W = self.weights.view(self.Z, self.L, self.F)  # Z x L x F
        H_t = H.transpose(0, 1)  # F x L
        R = torch.zeros(self.Z, self.L)
        for i in range(self.Z):
            w_i = W[i]  # L x F
            r_i = w_i @ H_t  # L x L
            r_i = r_i.diag()  # L
            R[i] = r_i
        Z = Z.detach()  # allows numpy conversion
        Z = Z.unsqueeze(0)  # allows to be treated as a 2D
        R = F.softmax(R, dim=1)
        R = R.detach()
        return Z, R  # 1 x Z, Z x L


TOKENIZER = AutoTokenizer.from_pretrained('roberta-base')
pretrained = AutoModel.from_pretrained(config.from_pretrained)
MODEL = Twin.from_pretrained('./trained_twin_96', pretrained)
VIZ = LatentVis(MODEL)

TITLE = """Damage-induced regeneration of the intestinal stem cell pool through enteroblast mitosis in the Drosophila midgut"""
ABSTRACT = """Many adult tissues and organs including the intestine rely on resident stem cells to maintain homeostasis and regeneration. In mammals, the progenies of intestinal stem cells (ISCs) can dedifferentiate to generate ISCs upon ablation of resident stem cells. However, whether and how mature tissue cells generate ISCs under physiological conditions remains unknown. Here, we show that infection of the Drosophila melanogaster intestine with pathogenic bacteria induces entry of enteroblasts (EBs), which are ISC progenies, into the mitotic cycle through upregulation of epidermal growth factor receptor (EGFR)-Ras signaling. We also show that ectopic activation of EGFR-Ras signaling in EBs is sufficient to drive enteroblast mitosis cell autonomously. Furthermore, we find that the dividing enteroblasts do not gain ISC identity as a prerequisite to divide, and the regenerative ISCs are produced through EB mitosis. Taken together, our work uncovers a new role for EGFR-Ras signaling in driving EB mitosis and replenishing the ISC pool during fly intestinal regeneration, which may have important implications for tissue homeostasis and tumorigenesis in vertebrates."""

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
styles = {
    'pre': {
        'font-family': 'menlo, courier, monospace',
    },
    'title-input': {
        'width': '100%'
    },
    'abstract-input': {
        'width': '100%'
    },
}


app.layout = html.Div([
    dcc.Store(id='current-tokens-1'),
    dcc.Store(id='latent-variable-1'),
    dcc.Store(id='r-matrix-1'),
    dcc.Store(id='current-tokens-2'),
    dcc.Store(id='latent-variable-2'),
    dcc.Store(id='r-matrix-2'),
    html.Div(className='row', children=[
        html.Div(className="six columns", children=[
            html.H4("Title"),
            dcc.Input(
                id="sample-1",
                type="text",
                placeholder="type your sentence here...",
                value=TITLE,
                style=styles['title-input'],
            ),
            html.Button(
                "Submit",
                id="submit-sample-1",
            )
        ]),
        html.Div(className="six columns", children=[
            html.H4("Abstract"),
            dcc.Textarea(
                id="sample-2",
                # type="text",
                placeholder="type your sentence here...",
                value=ABSTRACT,
                style=styles['abstract-input'],
            ),
            html.Button(
                "Submit",
                id="submit-sample-2",
            )
        ]),
    ]),
    html.Div(className='row', children=[
        html.H4("Twin latent variable and their products:"),
        dcc.Graph(id='heatmap'),
    ]),
    html.Div(className='row', children=[
        html.Span("Selected index: "),
        html.Span(id='hover-data', style=styles['pre'])
    ]),
    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            html.H4("Highlights in title:"),
            html.P(id='highlights-1')
        ]),
        html.Div(className='six columns', children=[
            html.H4("Highlights in abstract:"),
            html.P(id='highlights-2')
        ]),
    ])
])


@app.callback(
    Output('current-tokens-1', 'data'),
    Output('r-matrix-1', 'data'),
    Output('latent-variable-1', 'data'),
    Input('submit-sample-1', 'n_clicks'),
    State('sample-1', 'value')
)
@app.callback(
    Output('current-tokens-2', 'data'),
    Output('r-matrix-2', 'data'),
    Output('latent-variable-2', 'data'),
    Input('submit-sample-2', 'n_clicks'),
    State('sample-2', 'value')
)
def get_input(n_clicks, sample):
    tokens = []
    if sample:
        tokenized = TOKENIZER(
            sample,
            return_tensors="pt",
            padding="max_length",
            max_length=config.max_length[0],
            truncation=True
        )
        input_ids = tokenized['input_ids']
        tokens = [TOKENIZER.decode(i) for i in input_ids[0] if i != MODEL.config.pad_token_id]
        Z, R = VIZ.contribution(tokenized)
        return tokens, R, Z
    else:
        return [], []


@app.callback(
    Output('heatmap', 'figure'),
    Input('latent-variable-1', 'data'),
    Input('latent-variable-2', 'data')
)
def show_latent_variables(z_1, z_2):
    z_1 = torch.tensor(z_1)
    z_2 = torch.tensor(z_2)
    z_3 = (z_1 * z_2)
    z = [z_1, z_2, z_3]
    heatmap = torch.cat(z, dim=0)
    fig = px.imshow(heatmap)
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(height=200, width=1200)
    return fig


@app.callback(
    Output('hover-data', 'children'),
    Input('heatmap', 'hoverData')
)
def display_hover_data(hoverData):
    idx = hoverData['points'][0]['x'] if hoverData else ""
    return idx


@app.callback(
    Output('highlights-1', 'children'),
    Input('heatmap', 'hoverData'),
    Input('current-tokens-1', 'data'),
    Input('r-matrix-1', 'data')
)
@app.callback(
    Output('highlights-2', 'children'),
    Input('heatmap', 'hoverData'),
    Input('current-tokens-2', 'data'),
    Input('r-matrix-2', 'data')
)
def highlight_token(hoverData, tokens, R):
    i = hoverData['points'][0]['x'] if hoverData else None
    if i is not None:
        R = torch.tensor(R)
        mean = R.mean()
        std = R.std()
        minimum = R.min()
        maximum = R.max()
        delta = maximum - minimum
        logging.info(f"R: {minimum}, {mean} +- {std}, {maximum}")
        spans = [
            html.Span(t, style={'color': f'rgb({(255 * ((R[i,j] - minimum) / delta))}, 0, 0)'}) if (R[i, j] > mean + std) or (R[i, j] < mean - std) else html.Span(t)
            for j, t in enumerate(tokens)
        ]
    else:
        spans = [html.Span("Nothing to show yet...")]
    return spans


if __name__ == '__main__':

    app.run_server(host="0.0.0.0", port=8050, use_reloader=True)
