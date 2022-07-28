# Aca hacemos el pipeline

# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

df_r1=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv")\

df_r2=pd.read_csv('../models/tmdb_5000_credits.csv')

# Vamos a hacer un merge entre los dos CV del de las movies con los credits r1 y r2 respectivamente

df_r1 = df_r1.merge(df_r2, on='title')

df_r1 = df_r1[['movie_id','title','overview','genres','keywords','cast','crew']]

df_r1.dropna(inplace = True)

# Creamos una funcion para convertir el objeto tratado

def convertir(objeto):
    lista = []
    for i in ast.literal_eval(objeto):
        lista.append(i['name'])
    return lista 

# Aplicamos la funcion a genres y keywords

df_r1['genres'] = df_r1['genres'].apply(convertir)

df_r1['keywords'] = df_r1['keywords'].apply(convertir)


# Definimos otra funcion para otra conversion

def convertir_2(objeto):
    lista = []
    contador = 0
    for i in ast.literal_eval(objeto):
        if contador < 3:
            lista.append(i['name'])
        contador +=1  
    return lista

# Aplicamos la funcion a cast

df_r1['cast'] = df_r1['cast'].apply(convertir_2)

# Definimos otra funcion

def director(objeto):
    lista = []
    for i in ast.literal_eval(objeto):
        if i['job'] == 'Director':
            lista.append(i['name'])
            break
    return lista 

# Aplicamos la funcion

df_r1['crew'] = df_r1['crew'].apply(director)

# Hacemos el cambio en overview

df_r1['overview'] = df_r1['overview'].apply(lambda x : x.split())

# Paso 7
# Usamos la funcion que esta en el paso 7

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

df_r1['cast'] = df_r1['cast'].apply(collapse)
df_r1['crew'] = df_r1['crew'].apply(collapse)
df_r1['genres'] = df_r1['genres'].apply(collapse)
df_r1['keywords'] = df_r1['keywords'].apply(collapse)

df_r1['tags'] = df_r1['overview']+df_r1['genres']+df_r1['keywords']+df_r1['cast']+df_r1['crew']

new_df = df_r1[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 ,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors).shape

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]

# Finalmente creamos la funcion que nos va a recomendar peliculas similares

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# Probamos ahora esa funcion
print()
print()
print(" Probando la funcion de recomendacion de peliculas")
print()
lista=['El Mariachi','Spectre','Pacific Rim','The Wolfman']
for i in lista:
    print(f'Recomendados para la Pelicula: {i}:')
    recommend(i)  
    print()
    print('>'*80)
    print()



print()
print('='*75)
print('        F  I  N   DEL   P R O G R A M A')
print('='*75)


