import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------------------
# Load Dataset
# ---------------------------

df = pd.read_csv(r"C:\Users\Gopi Reddy\Downloads\Hybrid dataset\myntradataset\styles.csv")

print(df.head())
print(df.isnull().sum())

# ---------------------------
# Content-Based Filtering
# ---------------------------

content_df = df[
    ['Product ID', 'Product Name', 'Brand',
     'Category', 'Color', 'Size']
].copy()

content_df['Content'] = content_df.apply(
    lambda row: ' '.join(row.astype(str)),
    axis=1
)

tfidf_vectorizer = TfidfVectorizer()

content_matrix = tfidf_vectorizer.fit_transform(
    content_df['Content']
)

content_similarity = linear_kernel(
    content_matrix,
    content_matrix
)

# ---------------------------
# Collaborative Filtering
# ---------------------------

reader = Reader(rating_scale=(1, 5))

surprise_data = Dataset.load_from_df(
    df[['User ID', 'Product ID', 'Rating']],
    reader
)

trainset = surprise_data.build_full_trainset()

algo = SVD()

algo.fit(trainset)

# ---------------------------
# Content Recommendations
# ---------------------------

def get_content_based_recommendations(
        product_id,
        top_n=10):

    if product_id not in content_df['Product ID'].values:
        return []

    index = content_df[
        content_df['Product ID'] == product_id
    ].index[0]

    similarity_scores = list(
        enumerate(content_similarity[index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similar_indices = [
        i[0]
        for i in similarity_scores[1:top_n+1]
    ]

    recommendations = content_df.iloc[
        similar_indices
    ]['Product ID'].tolist()

    return recommendations

# ---------------------------
# Collaborative Recommendations
# ---------------------------

def get_collaborative_filtering_recommendations(
        user_id,
        top_n=10):

    testset = trainset.build_anti_testset()

    user_testset = [
        row for row in testset
        if row[0] == str(user_id)
    ]

    predictions = algo.test(user_testset)

    predictions.sort(
        key=lambda x: x.est,
        reverse=True
    )

    recommendations = [
        int(pred.iid)
        for pred in predictions[:top_n]
    ]

    return recommendations

# ---------------------------
# Hybrid Recommendations
# ---------------------------

def get_hybrid_recommendations(
        user_id,
        product_id,
        top_n=10):

    content_recs = get_content_based_recommendations(
        product_id,
        top_n
    )

    collab_recs = get_collaborative_filtering_recommendations(
        user_id,
        top_n
    )

    hybrid_recs = list(
        dict.fromkeys(
            content_recs + collab_recs
        )
    )

    return hybrid_recs[:top_n]

# ---------------------------
# Example
# ---------------------------

user_id = 79
product_id = 5
top_n = 10

recommendations = get_hybrid_recommendations(
    user_id,
    product_id,
    top_n
)

print(
    f"\nHybrid Recommendations "
    f"for User {user_id}:"
)

for i, rec in enumerate(
        recommendations,
        start=1):

    print(
        f"{i}. Product ID: {rec}"
    )