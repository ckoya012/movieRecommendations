# Content-Based Movie Recommendation System

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions #######

def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

##################################################

## read CSV file
df = pd.read_csv("movie_dataset.csv")

## select features
features = ['keywords', 'cast', 'genres', 'director'] 

## create column in DF which combines all selected features

# loops through all features and modifies the data frame by filling all NaN with empty string
for feature in features:
        df[feature] = df[feature].fillna('')

def combine_features(row):
        try:
                return row['keywords'] +" "+ row['cast'] +" "+ row['genres'] +" "+ row['director']
        except:
                print ("Error: ", row)
                
df["combined_features"] = df.apply(combine_features, axis=1) #axis will pass each row individually and not columns

## count matrix from this new combined column
cv = CountVectorizer()

count_matrix = cv.fit_transform(df["combined_features"])

## cosine similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Avatar"

movie_index = get_index_from_title(movie_user_likes)

## find all similar movies in desc order of similarity score
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

## print first 50 movies
i = 0
for movie in sorted_similar_movies:
        print(get_title_from_index(movie[0]))
        i= i+1
        if (i > 50):
                break
