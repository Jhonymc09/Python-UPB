# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 10:28:46 2022

@author: Jhony Meléndez
"""

#Diplomado Python aplicado a la ingeniería
#Autor Jhony Meléndez
#ID 502198
#Jhony.melendez@upb.edu.co

# Tarea de clustering, enviar resultador por Git

# Tratamiento de datos
# ==============================================================================

#Importamos Librerias

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Importamos Librerias Gráficos
# ==============================================================================

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Importamos Librerias Preprocesado y modelado
# ==============================================================================
from sklearn.cluster import AgglomerativeClustering

# Con la clase Scikit-Learn se pueden entrenar modelos de clustering utilizando el algoritmo 
#hierarchical clustering aglomerativo.

from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

def plot_dendrogram(model, **kwargs):
    '''
    Esta función extrae la información de un modelo AgglomerativeClustering
    y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
    '''
    
    counts = np.zeros(model.children_.shape[0]) #Devuelve una nueva matriz de forma y tipo dados, llena de ceros.
    n_samples = len(model.labels_) 
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, 
                                      counts]).astype(float) #La función se usa para apilar arreglos 1-D como columnas en un arreglo 2-D

# Plot
    dendrogram(linkage_matrix, **kwargs)  # Mostrar la agrupación de datos
    
# Simulación de datos
# ==============================================================================
X, y = make_blobs(  # Nos ayuda a proporcionar un mayor control de los datos
        n_samples    = 200, 
        n_features   = 2, 
        centers      = 4, 
        cluster_std  = 0.60, 
        shuffle      = True, 
        random_state = 0
       )

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84)) # Nos facilita la creación de diseños comunes de subgráficos
for i in np.unique(y): # Nos permite enconntrar los elementos únicos de una matriz.
    ax.scatter(  # Nos ayuda a moistar gráfico de dispersión de Y frente a X con diferentes tamaños y colores.
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],   # Personalizacion
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
ax.set_title('Datos simulados')
ax.legend();

# Escalado de datos
# ==============================================================================
X_scaled = scale(X)  

# Modelos
# ==============================================================================
modelo_hclust_complete = AgglomerativeClustering(  
                            affinity = 'euclidean',
                            linkage  = 'complete',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_complete.fit(X=X_scaled)  #Acepta una entrada para los datos de muestra (X) 
                                        #y para modelos supervisados ​​también acepta un argumento para etiquetas de datos (Y)
modelo_hclust_average = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'average',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_average.fit(X=X_scaled)

modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            distance_threshold = 0,
                            n_clusters         = None
                     )
modelo_hclust_ward.fit(X=X_scaled)

# Dendrogramas
# ==============================================================================
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
plot_dendrogram(modelo_hclust_average, color_threshold=0, ax=axs[0])
axs[0].set_title("Distancia euclídea, Linkage average")
plot_dendrogram(modelo_hclust_complete, color_threshold=0, ax=axs[1])
axs[1].set_title("Distancia euclídea, Linkage complete")
plot_dendrogram(modelo_hclust_ward, color_threshold=0, ax=axs[2])
axs[2].set_title("Distancia euclídea, Linkage ward")
plt.tight_layout();

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
altura_corte = 6
plot_dendrogram(modelo_hclust_ward, color_threshold=altura_corte, ax=ax)
ax.set_title("Distancia euclídea, Linkage ward")
ax.axhline(y=altura_corte, c = 'black', linestyle='--', label='altura corte')
ax.legend();

# Método silhouette para identificar el número óptimo de clusters
# ==============================================================================
range_n_clusters = range(2, 15)
valores_medios_silhouette = []

for n_clusters in range_n_clusters:
    modelo = AgglomerativeClustering(
                    affinity   = 'euclidean',
                    linkage    = 'ward',
                    n_clusters = n_clusters
             )

    cluster_labels = modelo.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    valores_medios_silhouette.append(silhouette_avg)
    
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.plot(range_n_clusters, valores_medios_silhouette, marker='o')
ax.set_title("Evolución de media de los índices silhouette")
ax.set_xlabel('Número clusters')
ax.set_ylabel('Media índices silhouette');

# Modelo
# ==============================================================================
modelo_hclust_ward = AgglomerativeClustering(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            n_clusters = 4
                     )
modelo_hclust_ward.fit(X=X_scaled)