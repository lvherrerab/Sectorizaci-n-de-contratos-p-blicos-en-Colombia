
# importar librerias

# librerias base
import os

# librerias externas
import pandas as pd
from sodapy import Socrata

import os
import sys

import pandas as pd
import numpy as np
import unidecode
import re
import spacy

# librerias propias
dir_proyecto, _ = os.path.split(os.getcwd())
sys.path.append(dir_proyecto)

import os
import sys
import re
import spacy
import matplotlib.pyplot as plt
import nltk
from nltk.stem import SnowballStemmer
import unidecode
import textwrap

# contexto
#import contexto
#from contexto.limpieza import limpieza_basica, limpieza_texto, remover_acentos, lista_stopwords, remover_palabras_cortas, quitar_repetidos
#from contexto.correccion import Corrector, corregir_texto
#from contexto.stemming import Stemmer, stem_texto

#from contexto.lematizacion import LematizadorSpacy, LematizadorStanza
#from contexto.lematizacion import lematizar_texto

# from contexto.limpieza import * #limpieza_basica
# from contexto.correccion import Corrector, corregir_texto

# lista_stop_contexto = []
# for i in lista_nombres()[0]:
#     lista_stop_contexto.append(i.lower())

# for i in lista_apellidos()[0]:
#     lista_stop_contexto.append(i.lower())

# for i in lista_geo_colombia('municipios')[0]:
#     lista_stop_contexto.append(i.lower())

# for i in lista_geo_colombia('departamentos')[0]:
#     lista_stop_contexto.append(i.lower())

# pd.DataFrame(set(lista_stop_contexto)).to_csv('../INPUTS/NLP/contexto_stopwords.csv',header=False,index=False)


# Inicializar lematizador para lenguaje español
lematizador = LematizadorStanza('es')

# # Cargamos las stopwords extra
nlp = spacy.load("es_core_news_sm")
extra_stopwords = pd.concat([pd.read_csv('../INPUTS/NLP/extra_stopwords.csv', sep=',',header=None),pd.read_csv('../INPUTS/NLP/contexto_stopwords.csv', sep=',',header=None)])
extra_stopwords.columns = ['stopwords']
extra_stopwords=set(extra_stopwords['stopwords'].to_list())
nlp.Defaults.stop_words |= extra_stopwords



def Lematizador_propio(text):
    nlp = spacy.load("es_core_news_sm")
    # Diccionario con las palabras y sus lemas correspondientes
    lemmas = {
        r"\babandona\b": "abandonar",
        r"\babandonado\b": "abandonar",
        r"\babandonandolo\b": "abandonar",
        r"\babandonar\b": "abandonar",
        r"\babandono\b": "abandonar",
        r"\bzombie\b": "zombie",
        r"\bzombi\b": "zombie",
        r"\bzombieland\b": "zombie",
        r"\bzombis\b": "zombie",
        r"\bzoo\b": "zoo",
        r"\bzoologo\b": "zoo"
    }

    # Buscar y reemplazar las palabras usando expresiones regulares
    for pattern, lemma in lemmas.items():
        text = re.sub(pattern, lemma, text, flags=re.IGNORECASE)

    #NLP object
    text = nlp(text)
    #Eliminar palabras de parada 'stopwords'
    text = [token.text if len(set(token.text))>1 else " " for token in text if not token.is_stop]
    text = " ".join(text)    
    text = re.sub(r"\s+", " ", text)

    ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
    for letter in ascii_lowercase:
        text = re.sub("\s+".format, " ", text)


    return text


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
    out = [token.text for token in nlp(out) if len(token) > 2]

    return out


def limpiar_texto_mas_stemming(texto):
    nlp = spacy.load("es_core_news_sm")
    texto = " ".join(limpiar_texto(texto))
    # Aplicar stemming
    spanish_stemmer = SnowballStemmer('spanish')
    out = nlp(texto)
    out = [spanish_stemmer.stem(token.text) for token in nlp(out)]

    return out

def limpieza_contexto(string):

    texto = str(string)
    # Limpieza básica, se pasa todo a minúsculas, se eliminan signos de puntuación y caracteres numéricos
    texto = limpieza_basica(texto)
    
    ascii_lowercase = 'abcdefghijkmnopqstuvwxyz' 
    for letter in ascii_lowercase:
        texto = re.sub(r"{}+".format(letter), letter, texto)

    ascii_lowercase = 'rl' 
    for letter in ascii_lowercase:
        texto = re.sub(r"{}{{2,}}".format(letter+letter), letter, texto)

    texto = remover_acentos(texto)
    texto = remover_palabras_cortas(texto, 3)
    texto = quitar_repetidos(texto)
    texto = corregir_texto(texto)
    texto = remover_acentos(texto)
    texto = limpieza_texto(texto)

    #Algunas veces se ponen tildes en el paso de limpieza_texto
    texto = limpieza_basica(texto)
    #Eliminar espacios seguidos'    

    # eliminar palabras que superen 23 letras, la palabra mas larga de español es 'electroencefalografista'

    texto = [token.text  for token in nlp(texto) if len(token.text) <= 23]
    texto = " ".join(texto)
   
    texto = [token.text for token in nlp(texto) if not token.is_stop]
    texto = " ".join(texto)    
    texto = re.sub(r"\s+", " ", texto)

    #texto = lematizar_texto(texto, libreria='stanza')
    texto = lematizador.lematizar(texto)
    # texto = lematizar_texto(texto, lematizador=LematizadorSpacy('es'))
    texto = remover_acentos(texto)
    texto = [token.text for token in nlp(texto)]

    return texto

def limpieza_contexto_stemming(string):
    texto = " ".join(limpieza_contexto(string))
    texto = stem_texto(texto)
    texto = limpieza_basica(texto)
    texto = [token.text for token in nlp(texto)]
    return texto

# definir funciones

def cliente_socrata(domain="www.datos.gov.co",token='grLe4ysf5HvZXuFKeZJlGZL7b'):
    """
    

    Parameters
    ----------
    domain : TYPE, optional
        DESCRIPTION. The default is "www.datos.gov.co".
    token : TYPE, optional
        DESCRIPTION. The default is 'grLe4ysf5HvZXuFKeZJlGZL7b'.

    Returns
    -------
    client : TYPE
        DESCRIPTION.

    """
    client = Socrata(domain,token)
    return client

def consultar_conjunto(id_conjunto, limit, ordenar):
    """
    

    Parameters
    ----------
    id_conjunto : TYPE
        DESCRIPTION.
    limit : TYPE
        DESCRIPTION.
    ordenar : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    client = cliente_socrata()
    results = client.get(id_conjunto, limit=limit, order=ordenar)
    results_df = pd.DataFrame.from_records(results)
    results_df.to_parquet(f"Conjunto_{id_conjunto}.parquet.gzip")


def consultar_metadata(id_conjunto):
    """
    

    Parameters
    ----------
    id_conjunto : TYPE
        DESCRIPTION.

    Returns
    -------
    columns : TYPE
        DESCRIPTION.

    """
    client = cliente_socrata()
    metadatos = client.get_metadata(id_conjunto)
    columns = [x["fieldName"] for x in metadatos["columns"]]
    return columns

def consultar_conjunto_paginado(id_conjunto, limit, ordenar, offset, **kwards):
    """
    

    Parameters
    ----------
    id_conjunto : TYPE
        DESCRIPTION.
    limit : TYPE
        DESCRIPTION.
    ordenar : TYPE
        DESCRIPTION.
    offset : TYPE
        DESCRIPTION.
    **kwards : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    where = kwards.get('where', '')
    client = cliente_socrata()
    metadatos = consultar_metadata(id_conjunto)
    results = client.get(id_conjunto, limit=limit, order=ordenar, offset=limit*offset, where =where)
    results_df = pd.DataFrame.from_records(results)#
    results_df.reindex(columns=metadatos)
    results_df.to_parquet(f"../INPUTS/Conjunto_{id_conjunto}_{offset}.parquet.gzip")

def leer_parquet(carpeta,extension='parquet.gzip'):
    """
    

    Parameters
    ----------
    carpeta : TYPE
        DESCRIPTION.
    extension : TYPE, optional
        DESCRIPTION. The default is 'parquet.gzip'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    archivos_parquet = [f for f in os.listdir(f'{carpeta}') if extension in f]
    data_frames = []
    for archivo in archivos_parquet:
        data_frames.append(pd.read_parquet(os.path.join(carpeta,archivo)))
    return pd.concat(data_frames)

def dimensionar_conjunto(id_conjunto, **kwargs):
    """
    

    Parameters
    ----------
    id_conjunto : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    where = kwargs.get('where', '')
    client = cliente_socrata()
    conteo = client.get(id_conjunto, select = "count(1)", where = where)
    results_df = pd.DataFrame.from_records(conteo)
    return int(results_df.iloc[0,0])



def lematizado_contratos(texto):

    stop_words = ['contrato', 'departamento', 'municipio']

    texto = nlp(texto)
    #Eliminar palabras de parada 'stopwords'
    texto = [token.text for token in texto if token.text not in stop_words]
    texto = " ".join(texto)  
    # Corregir errores
    lemmas = {
        r"\b\w*construction\w*\b": "construir",        
        r"\b\w*suminist\w*\b": "suministrar",
        r"\b\w*servici\w*\b": "servicio",
        r"\b\w*adecua\w*\b": "adecuar",
        r"\b\w*abastec\w*\b": "abastecer",
        r"\b\w*acompan\w*\b": "acompanar",
        r"\b\w*acondic\w*\b": "acondicionar",
        r"\b\w*activi\w*\b": "activar",
        r"\b\w*actual\w*\b": "actual",
        r"\b\w*administ\w*\b": "administrar",
        r"\b\w*adop\w*\b": "adoptar",
        r"\b\w*aliment\w*\b": "alimentar",
        r"\b\w*anali\w*\b": "analizar",
        r"\b\w*aproba\w*\b": "aprobar",
        r"\b\w*arrenda\w*\b": "arrendar",
        r"\b\w*articul\w*\b": "articular",
        r"\b\w*asegurar\w*\b": "asegurar",
        r"\b\w*aseso\w*\b": "asesorar",
        r"\b\w*automa\w*\b": "automatizar",
        r"\b\w*benefic\w*\b": "beneficiar",
        r"\b\w*calif\w*\b": "calificar",
        r"\b\w*ciudada\w*\b": "ciudadano",
        r"\b\w*colabora\w*\b": "colaborar",
        r"\b\w*comercia\w*\b": "comerciar",
        r"\b\w*complej\w*\b": "complejo",
        r"\b\w*complement\w*\b": "complementar",
        r"\b\w*ejecut[^i]\w*\b": "ejecutar",
        r"\b\w*conforma\w*\b": "conformar",
        r"\b\w*conmemo\w*\b": "conmemorar",
        r"\b\w*desarrolla\w*\b": "desarrollar",
        r"\b\w*presupuest\w*\b": "presupuestar",
        r"\b\w*product\w*\b": "producir",
        r"\b\w*reconoci\w*\b": "reconocer",
        r"\b\w*reconst\w*\b": "reconstruir",
        r"\b\w*reforest\w*\b": "reforestar",
        r"\b\w*renova[dc]\w*\b": "renovar",
        r"\b\w*salvaguar[dc]\w*\b": "salvaguardar",
        r"\b\w*seleccion\w*\b": "seleccionar",
        r"\b\w*tecnolog\w*\b": "tecnologia",
        r"\b\w*territori\w*\b": "territorio",
        r"\b\w*transforma\w*\b": "transformar",
        r"\b\w*vigilan\w*\b": "vigilar",
    }

    for pattern, lemma in lemmas.items():
        texto = re.sub(pattern, lemma, texto, flags=re.IGNORECASE)
    

    # Reemplazar sinónimos
    lemmas = {
        r"\bprestacion\b": "prestar",
        r"\badelantar\b": "desarrollar",
        r"\bejecutar\b": "desarrollar",
    }
    
    for pattern, lemma in lemmas.items():
        texto = re.sub(pattern, lemma, texto, flags=re.IGNORECASE)

    return texto



# =============================================================================
# PRUEBAS
# =============================================================================

# dimensionar_conjunto(id_conjunto)
# cliente = cliente_socrata()

# consultar_conjunto('p6dx-8zbt', 1000, "id_del_proceso")
# carpeta = r'D:\01_MaestriaAnalitica_UnivAndes\10_aprendizaje_no_supervisado\ans_proyecto\OUTPUTS\RAW'
# id_conjunto = 'p6dx-8zbt'
# limit = 1000
# leer_parquet()
# ordenar = "id_del_proceso"
# offset = 0

# leer_parquet(carpeta)
# archivo =   'Conjunto_p6dx-8zbt_0.parquet.gzip'
# pd.read_parquet(r"D:\01_MaestriaAnalitica_UnivAndes\10_aprendizaje_no_supervisado\ans_proyecto\OUTPUTS\RAW\Conjunto_p6dx-8zbt_0.parquet.gzip")
# pd.read_parquet(r"D:\01_MaestriaAnalitica_UnivAndes\10_aprendizaje_no_supervisado\ans_proyecto\Conjunto_p6dx-8zbt.parquet.gzip")

# consultar_conjunto_paginado(id_conjunto, limit, ordenar, offset)
