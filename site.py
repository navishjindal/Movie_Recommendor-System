import pickle
import streamlit as st
import requests
import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# --- Data Preprocessing ---
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert2(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def remove_space(L):
    return [i.replace(" ", "") for i in L]

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

# Load and process data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['genres', 'id', 'keywords', 'title', 'overview', 'production_companies', 'cast', 'crew']]
movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert2)
movies['crew'] = movies['crew'].apply(director)
movies['production_companies'] = movies['production_companies'].apply(convert)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
movies['production_companies'] = movies['production_companies'].apply(remove_space)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] + movies['production_companies']
movies_final = movies[['id', 'title', 'tags']]
movies_final['tags'] = movies_final['tags'].apply(lambda x: " ".join(x))
movies_final['tags'] = movies_final['tags'].apply(lambda x: x.lower())
movies_final['tags'] = movies_final['tags'].apply(stem)

# --- Vectorize and compute similarity ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies_final['tags']).toarray()
similarity = cosine_similarity(vector)

movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
movie_dict = pickle.load(open('movie_dict.pkl','rb'))

def fetch_poster(movie_id):
    api_key = 'your_tmdb_api_key'  # Replace this with your real TMDB API key
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhZGQyMjIwMjdhOTQwNWJkZDMxM2JiZDc2ZWViN2FlZiIsIm5iZiI6MTc1MjQyNDgwOC4wOTcsInN1YiI6IjY4NzNlMTY4N2FhYjE1MGZiOTcwNTYwZCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.VMilAyoISYAxTp2yeqaMURqvazBobfzLzyzlIxwhjq4'}&language=en-US"
    data = requests.get(url).json()



def recommend(movie):
    index = movies_final[movies_final['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(movies_final.iloc[i[0]].title)
    return recommended_movie_names


st.header('Movie Recommender System')

movie_list = movies_final['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
    with col2:
        st.text(recommended_movie_names[1])
    with col3:
        st.text(recommended_movie_names[2])
    with col4:
        st.text(recommended_movie_names[3])
    with col5:
        st.text(recommended_movie_names[4])