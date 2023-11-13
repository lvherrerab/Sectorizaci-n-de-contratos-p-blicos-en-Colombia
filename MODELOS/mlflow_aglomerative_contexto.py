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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

#
# Cargar data

df= pd.read_csv("https://raw.githubusercontent.com/lvherrerab/Sectorizaci-n-de-contratos-p-blicos-en-Colombia/main/OUTPUT/muestra_al_35_sector.csv")

#Preprocesamiento:

# Crear un vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Calcular la matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df['limpieza_contexto'])
matriz = tfidf_matrix.toarray()

# registre el experimento
experiment = mlflow.set_experiment("/Shared/Aglomerative")

# Aquí se ejecuta MLflow sin especificar un nombre o id del experimento. MLflow los crea un experimento para este cuaderno por defecto y guarda las características del experimento y las métricas definidas. 
# Para ver el resultado de las corridas haga click en Experimentos en el menú izquierdo. 
with mlflow.start_run(experiment_id=experiment.experiment_id):
    
    # defina los parámetros del modelo
    n_clusters=24
    affinity= 'cosine'
    linkage='ward'
    distance_threshold=0.5
    
    

    # Cree el modelo con los parámetros definidos y entrénelo
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage = linkage, distance_threshold= distance_threshold)
    agglomerative.fit(matriz)

    
    # Realice predicciones de prueba
    # Asignar etiquetas de cluster a cada documento
    labels = agglomerative.labels_
    print(labels)
  
    # Registre los parámetros
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("affinity", affinity)
    mlflow.log_param("linkage", linkage)
    mlflow.log_param("distance_threshold", distance_threshold)
  
    # Registre el modelo
    mlflow.sklearn.log_model(agglomerative, "agglomerative_cluster")
  
    # Cree y registre la métrica de interés
    
    silhouette_avg = silhouette_score(matriz, agglomerative.labels_)
    mlflow.log_metric("silhouette_avg", silhouette_avg)
    print("Silhouette Score del modelo Aglomerative es: ", silhouette_avg)
