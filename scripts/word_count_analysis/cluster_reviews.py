import os
import sys

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_clustering(tfidf_matrix: csr_matrix, num_clusters: int) -> pd.array:
    """
    Perform K-means clustering on a given TF-IDF matrix with a specific number of clusters.

    TF-IDF (Term Frequency and Inverse Document Frequency) uses the relative frequency of words
    to determine how relevant those words are to a given document.
    """

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(tfidf_matrix)

    return kmeans.predict(tfidf_matrix)


def main(input_csv: str, num_clusters: int, output_csv: str, overwrite: bool = False) -> None:
    """
    Main function to load preprocessed data, vectorize the 'Review_Text' column,
    perform K-means clustering, assign clusters to the original data, and save the updated data.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    if os.path.exists(output_csv) and not overwrite:
        should_proceed = input(f"The output file '{output_csv}' already exists. Do you want to overwrite it? (y/n): ")
        if should_proceed.lower() != 'y':
            sys.exit(f"Execution stopped. The output file '{output_csv}' already exists.")

    df = pd.read_csv(input_csv)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Review_Text'].values.astype('U'))
    cluster_assignments = tfidf_clustering(tfidf_matrix, num_clusters)

    df['Cluster'] = cluster_assignments
    df = df.sort_values(by='Cluster')
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    input_csv = './output/word_count_analysis/csv/preprocessed_reviews.csv'
    num_clusters = 9
    output_csv = f'./output/word_count_analysis/csv/reviews_with_{num_clusters}_clusters.csv'
    overwrite = False

    main(input_csv, num_clusters, output_csv, overwrite)
