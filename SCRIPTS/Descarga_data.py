#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:10:21 2023

@author: linaherrera
"""
#Librerias importas
import math
import os
import shutil 
import sys

#librerias externas
import pandas as pd 
import numpy as np
from sodapy import Socrata



dir_proyecto, _ = os.path.split(os.getcwd())
sys.path.append(dir_proyecto)


from funciones import *


# Carpetas del proyecto



try:
    os.mkdir(os.path.join(dir_proyecto,'INPUTS'))
except:
    None

try:
    shutil.rmtree(os.path.join(dir_proyecto,'INPUTS'))
except:
    None
os.mkdir(os.path.join(dir_proyecto,'INPUTS'))

# parámetros de las funciones

id_conjunto = 'jbjy-vk9h'
limit = 5000
ordenar = "id_contrato"
# filtro sobre el conjunto. Año 2023 cuyo valor sea superior a cien millones de pesos
where = "fecha_de_firma>='2023-01-01T00:00:00.000' and valor_del_contrato >= 100000000"


tamanio_conjunto = dimensionar_conjunto(id_conjunto, where=where)

loops = math.ceil(tamanio_conjunto/limit)

for offset in range (loops):
    consultar_conjunto_paginado(id_conjunto, limit, ordenar, offset, where = where)



contratacion = leer_parquet(os.path.join(dir_proyecto,'INPUTS'))
data = pd.DataFrame(contratacion)


data.to_csv('data.csv', index=False)