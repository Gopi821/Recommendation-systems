# Recommendation System Using Machine Learning

## Project Overview

This project demonstrates three popular recommendation system techniques used in modern applications such as Netflix, Amazon, Spotify, YouTube, and E-commerce platforms.

The project consists of:

1. Content-Based Filtering
2. Collaborative Filtering
3. Hybrid Recommendation System

The objective is to recommend relevant items to users based on item features, user preferences, and historical interactions.

---

## 1. Content-Based Filtering

Content-Based Filtering recommends items similar to those a user has liked in the past.

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TF-IDF Vectorization
* Cosine Similarity

### Dataset

* TMDB 5000 Movie Dataset

### Features Used

* Genres
* Keywords
* Tagline
* Cast
* Director

### Workflow

1. Load movie metadata.
2. Combine important movie features.
3. Convert text into numerical vectors using TF-IDF.
4. Calculate similarity scores using Cosine Similarity.
5. Recommend movies similar to the user's selected movie.

---

## 2. Collaborative Filtering

Collaborative Filtering recommends items based on the behavior and preferences of similar users.

### Technologies Used

* Python
* Pandas
* NumPy
* Pearson Correlation

### Dataset

* MovieLens Dataset

### Workflow

1. Build a User-Item Interaction Matrix.
2. Analyze user ratings.
3. Calculate correlations between movies.
4. Identify similar user preferences.
5. Recommend movies based on collective user behavior.

### Advantages

* Does not require item metadata.
* Learns directly from user interactions.
* Widely used in real-world recommendation systems.

---

## 3. Hybrid Recommendation System

Hybrid Recommendation Systems combine Content-Based and Collaborative Filtering techniques to improve recommendation quality.

### Technologies Used

* Python
* Pandas
* Scikit-Learn
* Surprise Library (SVD)
* TF-IDF Vectorization

### Dataset

* Myntra Fashion Products Dataset

### Workflow

1. Generate content-based recommendations using product attributes.
2. Generate collaborative recommendations using user ratings.
3. Merge both recommendation lists.
4. Produce more accurate and personalized recommendations.

### Benefits

* Reduces cold-start problems.
* Improves recommendation accuracy.
* Combines strengths of multiple recommendation approaches.

---

## Project Structure

```text
Recommendation-System/
│
├── Content_based_filtering.py
├── Collaborative_filtering.py
├── Hybrid_filtering.py
├── requirements.txt
├── README.md
└── data/
```

## Required Libraries

* pandas
* numpy
* scikit-learn
* scipy
* scikit-surprise

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

### TMDB 5000 Movie Dataset

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

### MovieLens Dataset

https://grouplens.org/datasets/movielens/latest/

### Myntra Fashion Dataset

https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog

## Learning Outcomes

Through this project, the following concepts were implemented:

* Natural Language Processing (NLP)
* Feature Engineering
* TF-IDF Vectorization
* Cosine Similarity
* User-Based Recommendation
* Item-Based Recommendation
* Matrix Factorization (SVD)
* Hybrid Recommendation Techniques

## Author

Gopi Pingali

Aspiring AI/ML Engineer passionate about Machine Learning, Recommendation Systems, NLP, and Generative AI.
