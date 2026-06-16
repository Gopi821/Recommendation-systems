# Importing libraries
import numpy as np
import pandas as pd
import difflib
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load datasets
# -------------------------------
movies_data = pd.read_csv(
    r"C:\Users\Gopi Reddy\Downloads\archive\tmdb_5000_movies.csv"
)

credits_data = pd.read_csv(
    r"C:\Users\Gopi Reddy\Downloads\archive\tmdb_5000_credits.csv"
)

# Merge datasets
movies = movies_data.merge(credits_data, on="title")

print("Dataset Shape:", movies.shape)

# -------------------------------
# Helper Functions
# -------------------------------

def convert_names(obj):
    """
    Extract names from JSON-like columns
    """
    names = []

    try:
        obj = ast.literal_eval(obj)

        for item in obj:
            names.append(item["name"])

    except:
        pass

    return " ".join(names)


def get_director(obj):
    """
    Extract director from crew column
    """
    try:
        obj = ast.literal_eval(obj)

        for item in obj:
            if item["job"] == "Director":
                return item["name"]

    except:
        pass

    return ""


# -------------------------------
# Feature Engineering
# -------------------------------

movies["genres"] = movies["genres"].apply(convert_names)
movies["keywords"] = movies["keywords"].apply(convert_names)
movies["cast"] = movies["cast"].apply(convert_names)
movies["director"] = movies["crew"].apply(get_director)

selected_features = [
    "genres",
    "keywords",
    "tagline",
    "cast",
    "director"
]

for feature in selected_features:
    movies[feature] = movies[feature].fillna("")

# Combine features
combined_features = (
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["tagline"] + " " +
    movies["cast"] + " " +
    movies["director"]
)

# -------------------------------
# TF-IDF Vectorization
# -------------------------------

print("\nCreating feature vectors...")

vectorizer = TfidfVectorizer(stop_words="english")

feature_vectors = vectorizer.fit_transform(combined_features)

# -------------------------------
# Cosine Similarity
# -------------------------------

print("Calculating similarity matrix...")

similarity = cosine_similarity(feature_vectors)

print("Similarity Matrix Shape:", similarity.shape)

# -------------------------------
# User Input
# -------------------------------

movie_name = input(
    "\nEnter your favourite movie name: "
)

# List of titles
list_of_all_titles = movies["title"].tolist()

# Find closest movie match
find_close_match = difflib.get_close_matches(
    movie_name,
    list_of_all_titles,
    n=1
)

if len(find_close_match) == 0:
    print("\nMovie not found in dataset.")
    exit()

close_match = find_close_match[0]

print("\nClosest Match:", close_match)

# -------------------------------
# Recommendation Engine
# -------------------------------

index_of_movie = movies[
    movies["title"] == close_match
].index[0]

similarity_score = list(
    enumerate(similarity[index_of_movie])
)

sorted_similar_movies = sorted(
    similarity_score,
    key=lambda x: x[1],
    reverse=True
)

# -------------------------------
# Display Recommendations
# -------------------------------

print("\nMovies Suggested For You:\n")

count = 1

for movie in sorted_similar_movies:

    index = movie[0]

    title = movies.iloc[index]["title"]

    if count <= 30:
        print(f"{count}. {title}")
        count += 1
    else:
        break