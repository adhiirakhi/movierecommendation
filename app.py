import pandas as pd

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Display first few rows
print(movies.head())
print(ratings.head())
movie_data = pd.merge(ratings, movies, on='movieId')
print(movie_data.isnull().sum())
movie_data.dropna(inplace=True)
movie_data['rating'] = movie_data['rating'].astype(float)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.histplot(movie_data['rating'], bins=10, kde=False)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
top_movies = movie_data.groupby('title').size().sort_values(ascending=False).head(10)
print(top_movies)
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')
from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
def recommend_movies(user_id, num_recommendations=5):
    user_idx = user_id - 1  # Adjust for zero-indexed
    similarity_scores = user_similarity[user_idx]
    similar_users = user_movie_matrix.index[similarity_scores.argsort()[-num_recommendations-1:-1]]
    
    # Movies watched by similar users
   def recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        raise KeyError(f"Title '{title}' is not in indices.")
    # Rest of your code...
    return recommendations

print(recommend_movies(1, 5))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
movie_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_similar_movies(movie_title, num_recommendations=5):
    movie_idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(movie_similarity[movie_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movies = [movies['title'][i[0]] for i in similarity_scores[1:num_recommendations+1]]
    return similar_movies

print(recommend_similar_movies('Toy Story (1995)', 5))
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(movie_data, test_size=0.2, random_state=42)
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(predicted, actual):
    predicted = predicted[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(predicted, actual))

# Assuming you have a prediction matrix for test_data
# predicted_ratings = ...
# actual_ratings = ...

# print(rmse(predicted_ratings, actual_ratings))
import streamlit as st

st.title('Movie Recommendation System')

movie_name = st.text_input('Enter a movie name:')
if movie_name:
    recommendations = recommend_similar_movies(movie_name)
    st.write('Recommendations:')
    for rec in recommendations:
        st.write(rec)
