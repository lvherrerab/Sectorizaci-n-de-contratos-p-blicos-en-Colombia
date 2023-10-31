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


# librerias propias
dir_proyecto, _ = os.path.split(os.getcwd())
sys.path.append(dir_proyecto)

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
