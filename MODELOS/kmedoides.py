import mlflow
import mlflow.sklearn
from sklearn_extra.cluster import KMedoids
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

df= pd.read_csv("https://raw.githubusercontent.com/lvherrerab/Sectorizaci-n-de-contratos-p-blicos-en-Colombia/main/OUTPUT/muestra_al_35_sector.csv")

# Seleccionar un subconjunto aleatorio del 10%
#df_sample = df.sample(frac=0.3)
#df_sample

# Preprocesamiento

# Crear un vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Calcular la matriz TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(df['limpieza_contexto'])


# registre el experimento
experiment = mlflow.set_experiment("/Shared/K-medoides")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    
    
    # defina los parámetros del modelo
    n_clusters=24
    init='random'
    random_state=42   
    
    # Cree el modelo con los parámetros definidos y entrénelo
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    kmedoids.fit(tfidf_matrix)

    # Asignar etiquetas de cluster a cada documento
    labels = kmedoids.labels_
    print(labels)
    
    # Log de los parámetros del modelo
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("init", init)
    mlflow.log_param("random_state", random_state)
    

    # Registre el modelo
    mlflow.sklearn.log_model(kmedoids, "Kmedoides")
  
    # Cree y registre la métrica de interés
    
    
    inercia = kmedoids.inertia_
    mlflow.log_metric("inertia_", inercia)
    print("Inercia del modelo K-Medoides:", inercia)
    
    
    silhouette_avg = silhouette_score(tfidf_matrix, labels)
    mlflow.log_metric("silhouette_avg", silhouette_avg)
    print("Silhouette Score del modelo K-Medoides:", silhouette_avg)

