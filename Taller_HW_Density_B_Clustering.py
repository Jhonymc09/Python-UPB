# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 23:49:46 2022

@author: Jhony Meléndez
"""

# Tratamiento de datos
# ==============================================================================

# Importamos librerias
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Importamos librerias de Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn.cluster import DBSCAN # Realica la agrupación en clústeres de DBSCAN a partir de una matriz de vectores
                                   # datos que contienen grupos de densidad similar.
from sklearn.preprocessing import scale # Estandariza los datos a lo largo de cualquier eje.
                                        # Centro a la media y componente de escala a la varianza unitaria.
from sklearn.metrics import silhouette_score # Calcula el coeficiente de silueta medio de todas las muestras.

# Configuración warnings
# ==============================================================================
import warnings # Control de advertencia
warnings.filterwarnings('ignore') # advertencia está controlada por el filtro de advertencia

url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/' \
      + 'Estadistica-machine-learning-python/master/data/multishape.csv'

# Lectura de datos en la url
datos = pd.read_csv(url)
datos.head()

# Escalado de datos
# ==============================================================================
X = datos.drop(columns='shape').to_numpy() # Suelta etiquetas específicas de filas o columnas.
X_scaled = scale(X)

# Modelo
# ==============================================================================
modelo_dbscan = DBSCAN(
                    eps          = 0.2,
                    min_samples  = 5,
                    metric       = 'euclidean',
                )

modelo_dbscan.fit(X=X_scaled) # ajusta una función, f, a los datos.

# Clasificación
# ==============================================================================
labels = modelo_dbscan.labels_

fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5)) # crea una figura y una cuadrícula de 
                                                 # subparcelas con una sola llamada, 
                                                 # al tiempo que proporciona un control 
                                                 # razonable sobre cómo se crean las 
                                                 # parcelas individuales

ax.scatter(
    x = X[:, 0],
    y = X[:, 1], 
    c = labels,
    marker    = 'o',
    edgecolor = 'black'
)

# Los outliers se identifican con el label -1
ax.scatter(  # gráfico de dispersión de y frente a x con diferentes tamaños y/o colores de marcador.
    x = X[labels == -1, 0],
    y = X[labels == -1, 1], 
    c = 'red',
    marker    = 'o',
    edgecolor = 'black',
    label = 'outliers'
)

ax.legend()  # Coloque una leyenda en los ejes.
ax.set_title('Clusterings generados por DBSCAN'); # titulo

# Número de clusters y observaciones "outliers"
# ==============================================================================
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)

# Impresion
print(f'Número de clusters encontrados: {n_clusters}')
print(f'Número de outliers encontrados: {n_noise}')