#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:18:24 2023

@author: linaherrera
"""

# Importar librerías
#Importe MLFlow para registrar los experimentos, el regresor de bosques aleatorios y la métrica de error cuadrático medio
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#
# Cargar data

df= pd.read_csv("https://raw.githubusercontent.com/lvherrerab/Sectorizaci-n-de-contratos-p-blicos-en-Colombia/main/OUTPUT/muestra_sincont_todas_varia.csv")

#Preprocesamiento:

# Crear un vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Calcular la matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df['objeto_del_contrato'])


# registre el experimento
experiment = mlflow.set_experiment("/Shared/K-medias")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    
    # defina los parámetros del modelo
    n_clusters=24
    n_init=10
    random_state=123
    
    

    # Cree el modelo con los parámetros definidos y entrénelo
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(tfidf_matrix)

    
    # Realice predicciones de prueba
    # Asignar etiquetas de cluster a cada documento
    labels = kmeans.labels_
    print(labels)
  
    # Registre los parámetros
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("n_init", n_init)
    mlflow.log_param("random_state", random_state)
  
    # Registre el modelo
    mlflow.sklearn.log_model(kmeans, "Kmedias")
  
    # Cree y registre la métrica de interés
    
    
    inercia = kmeans.inertia_
    mlflow.log_metric("inertia_", inercia)
    print("Inercia del modelo K-Means:", inercia)
    
    
    silhouette_avg = silhouette_score(tfidf_matrix, labels)
    mlflow.log_metric("silhouette_avg", silhouette_avg)
    print("Silhouette Score del modelo K-Means:", silhouette_avg)
