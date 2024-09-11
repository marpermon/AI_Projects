#!/usr/bin/env python
# coding: utf-8

# # Proyecto 1: Inteligencia Artificial Aplicada a la Ingeniería Eléctrica

# ## Algoritmo 1: K-means

# In[1]:


# Importamos las librerías
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


# In[2]:


# Importamos los datos del anexo A
csv_file = 'AnexoA.csv'
datosA = pd.read_csv(csv_file)
print(datosA.info())
print(datosA.describe())


# In[3]:


# Importamos los datos del anexo b
csv_file = 'AnexoB.csv'
datosB = pd.read_csv(csv_file)
print(datosB.info())
print(datosB.describe())


# In[4]:


# Análisis exploratorio gráfica de los datos


fig = px.scatter_3d(datosA, 
                    x="Abonados", 
                    y="DPIR", 
                    z="FPI", 
                    title="Datos Anexo A")
fig.update_traces(
    textfont=dict(
        family="Arial",
        size=18,
        color='red' 
    ),
    marker=dict(
        size=5
    )
)
fig.show()                        


# In[5]:


# Análisis exploratorio gráfica de los datos
fig = px.scatter_3d(datosB, 
                    x="Abonados", 
                    y="DPIR", 
                    z="FPI", 
                    title="Datos Anexo B")
fig.update_traces(
    textfont=dict(
        family="Arial",
        size=18,
        color="crimson"  
    ),
    marker=dict(
        size=5
    )
)
fig.show()


# ***Escalamos los datos***

# In[6]:


data = datosA.select_dtypes(include=[int,float])
scaler = StandardScaler() #investigar qué hace
data = scaler.fit_transform(data)


# ### Método del codo

# In[7]:


km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,10))
 
visualizer.fit(data)        
visualizer.show()


# Visualizamos que el gráfico no converge, sino que sigue bajando indefinidamente. Se elige k = 4 porque la librería yellowbrick proporciona un gráfico que sugiere un valor de k adecuado.

# In[8]:


# Elegimos k = 4 a partir de la gráfica
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit(data)
y_means = kmeans.predict(data).astype(str)
y_means = y_means.reshape(133,1)
# quitamos escalamientos
centers = scaler.inverse_transform(kmeans.cluster_centers_) 

data_kmeans_elbow =  datosA.copy()
data_kmeans_elbow["Cluster"]= y_means


# In[9]:


fig = px.scatter_3d(data_kmeans_elbow, 
                    x="Abonados", 
                    y="DPIR", z="FPI", 
                    color="Cluster", 
                    title='Clústers hechos con K-means y método del codo')
fig.add_trace(go.Scatter3d(
    x=centers[:, 0],
    y=centers[:, 1],
    z=centers[:, 2],
    mode='markers',
    marker=dict(size=20, 
                color='yellow',
                symbol='diamond'),
    name='Centroids'
))

fig.update_traces(
    textfont=dict(
        family="Arial",
        size=15,
        color="crimson"  
    ),
    marker=dict(
        size=5,
    )
)
fig.show()


# #### Estadísticas

# In[10]:


data_kmeans_grouped = data_kmeans_elbow.groupby("Cluster").agg({"Abonados":['mean'],
                                    "DPIR":['mean'],
                                    "FPI":['mean'],
                                  "Empresa":[pd.Series.mode],
                                    "Cluster":'count'})
print(data_kmeans_grouped)


# In[11]:


# generamos gráficos más comprensibles
data_kmeans_grouped1 = data_kmeans_elbow.groupby("Cluster").agg({"Abonados":['mean'],
                                    "DPIR":['mean'],
                                    "FPI":['mean'],
                                  "Empresa":[pd.Series.mode]
                                    })

data_kmeans_grouped1['Abonados'] = data_kmeans_grouped1['Abonados']/100
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
data_kmeans_grouped1.plot(kind='bar',
                          ax=ax[0], 
                          xlabel='Cluster', 
                          ylabel='Valor Promedio')
ax[0].legend(['Abonados/100', 'DPIR', 'FPI'])

empresas=[data_kmeans_grouped1.Empresa.values[i][0] for i in range(len(data_kmeans_grouped1.Empresa.values))]
data_kmeans_grouped2 = data_kmeans_elbow.groupby("Cluster").agg({"Cluster":'count'})


formatter = [f"{i} \n {j} \n {l}" for i,j,l in zip(list(data_kmeans_grouped1.index),
                                                   empresas,
                                                   list(data_kmeans_grouped2.Cluster))]
ax[1].pie(data_kmeans_grouped2.Cluster, 
          labels=formatter, 
          labeldistance=1.2)
ax[1].set(title="Tamaño de cluster y empresa predominante")
fig.suptitle('Estadísticas de los clusters obtenidos con Kmeans y método del codo')
plt.show()


# ### Coeficiente silueta

# In[12]:


sc = {"Número de Clusters":[],
      "Coeficiente silueta":[]}

for k in range(2,9):
      kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
      label = kmeans.labels_
      sil_coeff = silhouette_score(data,label,metric = 'euclidean')
      sc["Número de Clusters"].append(k)
      sc["Coeficiente silueta"].append(sil_coeff)

fig = px.line(sc, x="Número de Clusters", y="Coeficiente silueta",
             title="Método del Coeficiente silueta", 
              width=600, 
              height=600)

fig.show()


# In[13]:


# Elegimos k = 2 a partir de la gráfica
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit(data)
y_means = kmeans.predict(data).astype(str)
y_means = y_means.reshape(133,1)
centers = scaler.inverse_transform(kmeans.cluster_centers_) #quitamos escalamientos

data_kmeans_sil =  datosA.copy()
data_kmeans_sil["Cluster"]= y_means


# In[ ]:





# In[14]:


fig = px.scatter_3d(data_kmeans_sil, 
                    x="Abonados", 
                    y="DPIR", z="FPI", 
                    color="Cluster", 
                    title='Clústers hechos con K-means y coeficiente silueta')
fig.add_trace(go.Scatter3d(
    x=centers[:, 0],
    y=centers[:, 1],
    z=centers[:, 2],
    mode='markers',
    marker=dict(size=20, 
                color='yellow',
                symbol='diamond'),
    name='Centroids'
))

fig.update_traces(
    textfont=dict(
        family="Arial",
        size=15,
        color="crimson"  
    ),
    marker=dict(
        size=5,
    )
)
fig.show()


# #### Estadísticas

# In[15]:


data_kmeans_grouped = data_kmeans_sil.groupby("Cluster").agg({"Abonados":['mean'],
                                    "DPIR":['mean'],
                                    "FPI":['mean'],
                                  "Empresa":[pd.Series.mode],
                                    "Cluster":'count'})
print(data_kmeans_grouped)


# In[16]:


# generamos gráficos más comprensibles
data_kmeans_grouped1 = data_kmeans_sil.groupby("Cluster").agg({"Abonados":['mean'],
                                    "DPIR":['mean'],
                                    "FPI":['mean'],
                                  "Empresa":[pd.Series.mode]
                                    })

data_kmeans_grouped1['Abonados'] = data_kmeans_grouped1['Abonados']/100
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
data_kmeans_grouped1.plot(kind='bar',
                          ax=ax[0], 
                          xlabel='Cluster', 
                          ylabel='Valor Promedio')
ax[0].legend(['Abonados/100', 'DPIR', 'FPI'])

empresas=[data_kmeans_grouped1.Empresa.values[i][0] for i in range(len(data_kmeans_grouped1.Empresa.values))]
data_kmeans_grouped2 = data_kmeans_sil.groupby("Cluster").agg({"Cluster":'count'})


formatter = [f"{i} \n {j} \n {l}" for i,j,l in zip(list(data_kmeans_grouped1.index),
                                                   empresas,
                                                   list(data_kmeans_grouped2.Cluster))]
ax[1].pie(data_kmeans_grouped2.Cluster, labels=formatter, labeldistance=1.2)
ax[1].set(title="Tamaño de cluster y empresa predominante")
fig.suptitle('Estadísticas de los clusters obtenidos con Kmeans y coeficiente silueta')
plt.show()


# ## Algoritmo 2: DBSCAN

# Mediante prueba y error se concluyó que los clusters del algoritmo DBSCAN convergen en 2, y la cantidad mínima de puntos que rodean a un core point necesaria para conseguirlo fue 8

# ### Obtención de eps/epsilon

# In[17]:


neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(data)
distances, indices = neighbors_fit.kneighbors(data)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title("Método del codo para épsilon", fontsize=15)
plt.ylabel("Distancias al 8th-Nearest Neighbor, ascendente")
plt.xlabel("Índice de cada punto")
plt.show()


# In[18]:


dbscan = DBSCAN(eps=0.75, min_samples=4)
dbscan.fit(data)
data_DBSCAN = pd.DataFrame(datosA.copy()) #datos sin escalar
data_DBSCAN["Cluster"] = dbscan.labels_.astype(str).tolist()
data_DBSCAN["Cluster"].replace({'-1':'1'}, inplace=True)
fig = px.scatter_3d(data_DBSCAN, x="Abonados", y="DPIR", z="FPI", 
                    color="Cluster", title='Clústers hechos con DBSCAN')

fig.update_traces(
    textfont=dict(
        family="Arial",
        size=18,
        color="crimson" 
    ),
    marker=dict(
        size=5
    )
)
fig.show()


# ### Estadísticas

# In[19]:


data_DBSCAN_grouped = data_DBSCAN.groupby("Cluster").agg({"Abonados":['mean'],
                                    "DPIR":['mean'],
                                    "FPI":['mean'],
                                  "Empresa":[pd.Series.mode],
                                    "Cluster":'count'})
print(data_DBSCAN_grouped)


# In[20]:


# generamos gráficos para entender mejor la data
data_DBSCAN_grouped1 = data_DBSCAN.groupby("Cluster").agg({"Abonados":['mean'],
                                    "DPIR":['mean'],
                                    "FPI":['mean'],
                                  "Empresa":[pd.Series.mode]
                                    })
data_DBSCAN_grouped1['Abonados'] = data_DBSCAN_grouped1['Abonados']/100

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
data_DBSCAN_grouped1.plot(kind='bar',
                          ax=ax[0], 
                          xlabel='Cluster', 
                          ylabel='Valor Promedio')
ax[0].legend(['Abonados/100', 'DPIR', 'FPI'])


empresas=[data_DBSCAN_grouped1.Empresa.values[i][0] for i in range(len(data_DBSCAN_grouped1.Empresa.values))]
data_DBSCAN_grouped2 = data_DBSCAN.groupby("Cluster").agg({"Cluster":'count'})

formatter = [f"{i} \n {j} \n {l}" for i,j,l in zip(list(data_DBSCAN_grouped1.index),
                                                   empresas,
                                                   list(data_DBSCAN_grouped2.Cluster))]
ax[1].pie(data_DBSCAN_grouped2.Cluster, labels=formatter)
ax[1].set(title="Tamaño de cluster y empresa predominante")
fig.suptitle('Estadísticas de los clusters obtenidos con DBSCAN')
plt.show()


# # Extra: Datos sin escalar

# ## K- means: Método del codo

# In[21]:


sin_escalar = datosA.select_dtypes(include=[int,float])
km = KMeans(random_state=42)
visualizer = KElbowVisualizer(km, k=(2,10))
 
visualizer.fit(sin_escalar)
visualizer.show()


# In[22]:


kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit(sin_escalar)
y_means = kmeans.predict(sin_escalar).astype(str)
y_means = y_means.reshape(133,1)
centers = kmeans.cluster_centers_ 
data_kmeans_elbow_sin_escalar =  datosA.copy()
data_kmeans_elbow_sin_escalar["Cluster"]= y_means


# In[23]:


data_kmeans_grouped1 = data_kmeans_elbow_sin_escalar.groupby("Cluster").agg({"Abonados":['mean'],
                                    "DPIR":['mean'],
                                    "FPI":['mean'],
                                  "Empresa":[pd.Series.mode]
                                    })

data_kmeans_grouped1['Abonados'] = data_kmeans_grouped1['Abonados']/100
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
data_kmeans_grouped1.plot(kind='bar',
                          ax=ax[0], 
                          xlabel='Cluster', 
                          ylabel='Valor Promedio')
ax[0].legend(['Abonados/100', 'DPIR', 'FPI'])

empresas=[data_kmeans_grouped1.Empresa.values[i][0] for i in range(len(data_kmeans_grouped1.Empresa.values))]
data_kmeans_grouped2 = data_kmeans_elbow.groupby("Cluster").agg({"Cluster":'count'})


formatter = [f"{i} \n {j} \n {l}" for i,j,l in zip(list(data_kmeans_grouped1.index),
                                                   empresas,
                                                   list(data_kmeans_grouped2.Cluster))]
ax[1].pie(data_kmeans_grouped2.Cluster, labels=formatter, labeldistance=1.2)
ax[1].set(title="Tamaño de cluster y empresa predominante")
fig.suptitle('Estadísticas de los clusters obtenidos con Kmeans y método del codo con datos sin escalar')
plt.show()


# In[24]:


fig = px.scatter_3d(data_kmeans_elbow_sin_escalar, 
                    x="Abonados", 
                    y="DPIR", 
                    z="FPI", 
                    color="Cluster", 
                    title='Clústers sin escalar hechos con K-means y método del codo')
fig.add_trace(go.Scatter3d(
    x=centers[:, 0],
    y=centers[:, 1],
    z=centers[:, 2],
    mode='markers',
    marker=dict(size=20, 
                color='yellow',
                symbol='diamond'),
    name='Centroids'
))

fig.update_traces(
    textfont=dict(
        family="Arial",
        size=15,
        color="crimson"  
    ),
    marker=dict(
        size=5,
    )
)
fig.show()

