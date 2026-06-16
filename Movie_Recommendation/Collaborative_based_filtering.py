import pandas as pd

# Load datasets
ratings = pd.read_csv(r"C:\Users\Gopi Reddy\Downloads\Collaborative data set\ratings.csv")
movies = pd.read_csv(r"C:\Users\Gopi Reddy\Downloads\Collaborative data set\movies.csv")

# Merge datasets
ratings = pd.merge(movies, ratings).drop(['genres', 'timestamp'], axis=1)

print("Dataset Shape:", ratings.shape)

# Create User-Movie Matrix
userRatings = ratings.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

print("Before:", userRatings.shape)

# Keep movies with at least 10 ratings
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0)

print("After:", userRatings.shape)

# Calculate Pearson Correlation
corrMatrix = userRatings.corr(method='pearson')

print(corrMatrix.head())


def get_similar(movie_name, rating):
    if movie_name not in corrMatrix.columns:
        print(f"Movie not found: {movie_name}")
        return pd.Series(dtype=float)

    similar_ratings = corrMatrix[movie_name] * (rating - 2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)

    return similar_ratings


# -----------------------------------
# Romantic User Example
# -----------------------------------

romantic_lover = [
    ("(500) Days of Summer (2009)", 5),
    ("Alice in Wonderland (2010)", 3),
    ("Aliens (1986)", 1),
    ("2001: A Space Odyssey (1968)", 2)
]

similar_movies = pd.DataFrame()

for movie, rating in romantic_lover:
    temp = get_similar(movie, rating)
    similar_movies = pd.concat(
        [similar_movies, temp.to_frame().T],
        ignore_index=True
    )

print("\nRecommendations for Romantic User:\n")

print(
    similar_movies.sum()
    .sort_values(ascending=False)
    .head(20)
)

# -----------------------------------
# Action User Example
# -----------------------------------

action = [
    ("Amazing Spider-Man, The (2012)", 5),
    ("Mission: Impossible III (2006)", 4),
    ("Toy Story 3 (2010)", 2),
    ("2 Fast 2 Furious (2003)", 4)
]

similar_movies = pd.DataFrame()

for movie, rating in action:
    temp = get_similar(movie, rating)

    similar_movies = pd.concat(
        [similar_movies, temp.to_frame().T],
        ignore_index=True
    )

print("\nRecommendations for Action User:\n")

print(
    similar_movies.sum()
    .sort_values(ascending=False)
    .head(10)
)