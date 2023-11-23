#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:08:31 2023

@author: linaherrera
"""

# main.py
from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from model import k_medias

# Configuración de Flask
app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='K-Means Clustering API', description='API for K-Means clustering')

# Definir modelos de entrada y salida
ns = api.namespace('predict', description='Text Preprocessing and K-Means Prediction')

text_input_model = api.model('TextInput', {
    'text': fields.String(required=True, description='Text to preprocess and predict')
})

cluster_output_model = api.model('ClusterOutput', {
    'cluster_label': fields.Integer(description='Predicted cluster label')
})

# Definir la ruta de la API
@ns.route('/')
class TextClusterApi(Resource):

    @api.expect(text_input_model)
    @api.marshal_with(cluster_output_model)
    def post(self):
        data = api.payload
        input_text = data['text']

        # Realizar el preprocesamiento del texto y predecir el cluster
        cluster_label = clustering_model.predict_cluster(input_text)

        return {'cluster_label': cluster_label}, 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
