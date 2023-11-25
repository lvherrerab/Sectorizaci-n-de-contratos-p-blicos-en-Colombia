# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:24:10 2023

@author: danid
"""

from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd
app = Dash(__name__)

import requests
import json
from dash.dependencies import Input, Output, State
from loguru import logger
import os

# Load data from csv
def load_data():
    d=pd.read_csv("https://raw.githubusercontent.com/lvherrerab/Sectorizaci-n-de-contratos-p-blicos-en-Colombia/main/OUTPUT/muestra_sincont_todas_varia.csv",low_memory=False)
    d=d[['sector', 'valor_del_contrato', 'objeto_del_contrato']]
    d_agr_rec=d.groupby(by="sector").sum()['valor_del_contrato']
    d_agr_con=d.groupby(by="sector").count()['objeto_del_contrato']
    d_def=pd.concat([d_agr_rec, d_agr_con], axis=1)
    return d_def    

# Cargar datos
data = load_data()


data["valor_del_contrato"]=round(data["valor_del_contrato"]/1000000000, 2)

fig_1 = px.bar(data, x=data.index, y="objeto_del_contrato",
               title="NÚMERO DE CONTRATOS Y RECURSOS POR <br> DELEGADA SECTORIAL", text_auto=True)

fig_1.update_layout(xaxis_title="SECTOR",
                    yaxis_title="NÚMERO DE CONTRATOS",
                    font_color="#52AD9C",
                    title_font_color="#52AD9C",
                    title_x=0.5)


fig_1.update_xaxes(visible=False, showticklabels=False)
fig_1.update_yaxes(showgrid=True, gridwidth=0.25, 
                   gridcolor='#9FFCDF')


fig_2 = px.bar(data, x=data.index, y="valor_del_contrato", text_auto=True)

fig_2.update_layout(xaxis_title="SECTOR",
                    yaxis_title="RECURSOS <br> EN MILES DE MILLONES",
                    font_color="#52AD9C",
                    title_font_color="#52AD9C",
                    title_x=0.5)

fig_2.update_xaxes(showgrid=True, gridwidth=0.25, 
                   gridcolor='#9FFCDF')

fig_2.update_yaxes(showgrid=True, gridwidth=0.25, 
                   gridcolor='#9FFCDF')

app.layout = html.Div([
    html.H4('SECTORIZACIÓN DE LA CONTRATACIÓN PÚBLICA EN COLOMBIA - 2023', 
            style={"font-family": "Helvetica",
                   "text-align": "left", 
                   "color": "blue", 
                   "margin-top": "50px", 
                   "margin-left": "10px", 
                   "font-size": "50px"}),
    html.H4('A continuación puede ingresar el objeto del contrato', 
            style={"font-family": "Helvetica",
                   "text-align": "left", 
                   "color": "gray", 
                   "font-size": "30px"}),
    
    dcc.Input(id="input", type="text", 
              style={"width":"75%" , 
                     "padding": "12px 20px",
                     "box-sizing": "border-box",
                     "border-radius": "4px",
                     "border": "2px solid blue", 
                     "background-color": "#A0B9C6", 
                     "color": "white"}), 
    html.Button('Realizar asignación de sector', id='button', n_clicks=0, style={"width":"10%",  "padding": "12px 20px"} ),
  
    dcc.Graph(id="graph_1", figure=fig_1,
            style={'width': '70%', 
                    'display': 'inline-block',
                    "margin-top":"30px"}),
    
    dcc.Graph(id="graph_2", figure=fig_2,
            style={'width': '70%', 
                    'display': 'inline-block',
                    }),

    html.Div(id='resultado-container')  # Nueva línea agregada
])


# Agregar el componente dcc.Interval
dcc.Interval(
    id='interval-component',
    interval=1 * 1000,  # en milisegundos
    n_intervals=0
),




# PREDICTION API URL 
api_url = os.getenv('API_URL')
api_url = "http://{}:5001/api/predict".format(api_url)


@app.callback(
    Output(component_id='resultado-container', component_property='children'),
    [Input('interval-component', 'n_intervals')],
    [State(component_id='button', component_property='n_clicks'),
     State(component_id='input', component_property='value')]
)



def update_output_div(n_clicks, input_text):
    myreq = {
        "inputs": [
            {
            "text": str(input_text)
            }
        ]
      }
    headers =  {"Content-Type":"application/json", "accept": "application/json"}

    # POST call to the API
    response = requests.post(api_url, data=json.dumps(myreq), headers=headers)
    data = response.json()["cluster_label"]
    logger.info("Response: {}".format(data))

    # Pick result to return from json format
    result = "El sector correspondiente al contrato es:"  + str(data)
    
    return result 

 

if __name__ == '__main__':
    logger.info("Running dash")
    app.run_server(debug=True)

