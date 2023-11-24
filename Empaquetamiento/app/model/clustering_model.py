#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:23:52 2023

@author: linaherrera
"""

# model/clustering_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import unidecode
import re
import spacy

# Cargar el modelo NLP de Spacy
nlp = spacy.load("es_core_news_sm")

# Cargar datos
df = pd.read_csv("https://raw.githubusercontent.com/lvherrerab/Sectorizaci-n-de-contratos-p-blicos-en-Colombia/main/OUTPUT/muestra_sincont_todas_varia.csv")

# Definir la función de limpieza de texto
def limpiar_texto(string):
    
    #Estandarizar acentuaciones
    out = unidecode.unidecode(string)

    #Poner en minúsculas
    out = out.lower()
    
    #Eliminar caracteres que no sean del alfabeto básico latino [A-Za-z0-9_] espacios y saltos de línea
    out = re.sub(r"[^\w\s]|\n", ' ', out)
    
    #Eliminar uno o varios dígitos seguidos
    out = re.sub("\d+", "", out)
    
    #Eliminar todos los espacios seguidos luego de reemplazar el texto
    out = re.sub('\s+', ' ', out)

    #NLP object
    out = nlp(out)
    #Eliminar palabras de parada 'stopwords'
    out = [token.text for token in out if not token.is_stop]
    out = " ".join(out)

    #Lematizar
    lemmas =[token.lemma_ for token in nlp(out)]
    # Convertir la lista de lemmas nuevamente a texto
    out = " ".join(lemmas)

    #Remover artículos (el/la/lo/etc.) o palabras de dos letras")
    out = " ".join([token.text for token in nlp(out) if len(token) > 2])

    return out
    

# Crear vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Calcular la matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df['objeto_del_contrato'])

# Crea y entrena el modelo K-Means
n_clusters = 24
n_init = 10
random_state = 123

kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
kmeans.fit(tfidf_matrix)

def predict_cluster(input_text):
    # Realizar el preprocesamiento del texto
    preprocessed_text = limpiar_texto(input_text)

    # Utilizar el vectorizador TF-IDF para convertir el texto a una matriz TF-IDF
    input_matrix = tfidf_vectorizer.transform([preprocessed_text])

    # Utilizar el modelo K-Means para predecir el cluster
    cluster_label = kmeans.predict(input_matrix)[0]

    return cluster_label
