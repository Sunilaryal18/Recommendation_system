import random

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from  sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel




def get_title_from_index2(df, index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title2(df, title):
    return df[df.title == title]["index"].values[0]


def get_films2(title):
    df1 = pd.read_csv("C:\\Users\\rmado\\OneDrive\\Desktop\\PageProject\\static\\tags.csv");
    df3 = pd.read_csv("C:\\Users\\rmado\\OneDrive\\Desktop\\PageProject\\static\\movies.csv");
    df3 = df3[:10000]
    del df1["userId"]
    del df1["timestamp"]
    df2 = pd.DataFrame()
    df2['movieId'] = df1['movieId']
    df2['tag'] = df1['tag'].astype(str)
    df2 = df2.groupby("movieId")["tag"].apply(' '.join)
    df4 = pd.merge(df2, df3, on="movieId")
    del df4['genres']
    df4['index'] = df4.index


    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df4["tag"])
    cosine_sim = cosine_similarity(count_matrix)
    movie_index = get_index_from_title2(df4, title)
    #print(df4[df4.index == movie_index])
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    result = []
    for movie in sorted_similar_movies[:10]:
         result.append(df4[df4.index == movie[0]]['title'])


    #print(result)
    return (result[1].values)[0]





def get_films3(title):
    pd.set_option('display.max_columns', None)
    movies = pd.read_csv("C:\\Users\\rmado\\OneDrive\\Desktop\\PageProject\\static\\movies.csv")
    ratings = pd.read_csv("C:\\Users\\rmado\\OneDrive\\Desktop\\PageProject\\static\\ratings.csv")
    dataset = pd.merge(movies[:10000], ratings[:100000], how='left', on='movieId')
    genres = []




    highest_number_of_rating = dataset.groupby('title')[['rating']].count()
    highest_number_of_rating = highest_number_of_rating.nlargest(10, 'rating')
    highest_number_of_rating.shape
    del dataset['timestamp']
    table = dataset.pivot_table(index='title', columns='userId', values='rating')
    table.shape
    #dataset.rating.value_counts().sort_values().plot(kind='barh')
    table = table.fillna(0)
    matrix = csr_matrix(table.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(matrix)
    NearestNeighbors(algorithm='brute', metric='cosine')


    i = 0;
    for find in table.index.values:
        if find == title:
            #print()
            break
        else:
            i = i +1;
    #print(i)
    #We are randomly choosing a a movie to generate recommendation for using KNN

    table.index[i]
    table.iloc[i, :]
    distances, indices = model_knn.kneighbors(table.iloc[i, :].values.reshape(1, -1), n_neighbors=6)

    #Generating recommendation using KNN for the selected movie
    cosine = linear_kernel(matrix, matrix)

    def get_recommendation(title):
        idx = i
        #print(idx)
        scores = list(enumerate(cosine[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:5]
        #print(scores)
        movie_indices = [i[0] for i in scores]
        #print(movie_indices)
        return table.iloc[movie_indices]

    df = get_recommendation(table.index[i])
    return (df.index.values)[0]


def get_films1(title):
    movies = pd.read_csv("C:\\Users\\rmado\\OneDrive\\Desktop\\PageProject\\static\\movies.csv")
    movies = movies[:10000]['title']
    return random.choice(movies)






#get_films2("Toy Story (1995)")