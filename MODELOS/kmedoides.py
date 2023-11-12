import mlflow
import mlflow.sklearn
from sklearn_extra.cluster import KMedoids
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#import matplotlib.pyplot as plt

df= pd.read_csv("https://raw.githubusercontent.com/lvherrerab/Sectorizaci-n-de-contratos-p-blicos-en-Colombia/main/OUTPUT/muestra_sincont_todas_varia.csv")

# Seleccionar un subconjunto aleatorio del 10%
df_sample = df.sample(frac=0.3)
df_sample

# Preprocesamiento

# Crear un vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Calcular la matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sample['objeto_del_contrato']+ ' ' + df_sample['ciudad'])

# Obtener los términos (palabras) del vocabulario
terms = tfidf_vectorizer.get_feature_names_out()

# Mostrar la matriz TF-IDF y el vocabulario
print(tfidf_matrix.toarray())
print(terms)

# Crear una instancia del modelo K-Medoides con el número deseado de clusters (K)
kmedoids = KMedoids(n_clusters=24)

# Ajustar el modelo K-Medoides a tus datos
kmedoids.fit(tfidf_matrix)

# Asignar etiquetas de cluster a cada documento
labels = kmedoids.labels_

# Puedes acceder a los medoides de los clusters si lo deseas
medoids = kmedoids.medoid_indices_

# Visualizar las etiquetas asignadas a cada documento
print(labels)

n_clusters = 24  # Número de clusters deseado
kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
df_sample['cluster'] = kmedoids.fit_predict(tfidf_matrix)

distancias = kmedoids.transform(tfidf_matrix)  # Obtiene las distancias de cada punto a los medoides
suma_distancias = distancias.min(axis=1).sum()  # Suma las distancias mínimas de cada punto

print("Suma de las distancias a los medoides:", suma_distancias)

from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(tfidf_matrix, df_sample['cluster'])
print(f"Coeficiente de Silhouette: {silhouette_avg}")

# registre el experimento
experiment = mlflow.set_experiment("/Shared/K-medoides")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Log de los parámetros del modelo
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("random_state", 42)

    # Log de los resultados del modelo
    mlflow.log_param("medoids_indices", medoids.tolist())
    mlflow.log_param("cluster_labels", labels.tolist())
    mlflow.log_param("suma_distancias", suma_distancias)
    mlflow.log_param("silhouette_score", silhouette_avg)

    # Log del modelo K-Medoides (esto puede no ser posible si el modelo no es serializable)
    try:
        mlflow.sklearn.log_model(kmedoids, "kmedoids_model")
    except Exception as e:
        print(f"Error al intentar registrar el modelo: {str(e)}")





