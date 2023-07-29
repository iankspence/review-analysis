import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_inertia(tfidf_matrix: csr_matrix, cluster_counts: range) -> list:
    """
    Calculate inertia for various number of cluster centers.
    """

    inertias = []
    for count in cluster_counts:
        kmeans = KMeans(n_clusters=count, random_state=0, n_init='auto')
        kmeans.fit(tfidf_matrix)
        inertias.append(kmeans.inertia_)

    return inertias


def plot_elbow_curve(cluster_counts: range, inertias: list, output_png: str) -> None:
    """
    Plot the elbow curve indicating the optimal number of clusters.
    """

    plt.figure(figsize=(6, 6))
    plt.plot(cluster_counts, inertias, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method showing the optimal number of clusters')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')


def main(input_csv: str, cluster_counts: range, output_png: str, overwrite: bool = False) -> None:
    """
    Main function to load preprocessed data, vectorize the 'Review_Text' column,
    perform elbow analysis and plot the resulting curve.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    if os.path.exists(output_png) and not overwrite:
        should_proceed = input(f"The output file '{output_png}' already exists. Do you want to overwrite it? (y/n): ")
        if should_proceed.lower() != 'y':
            sys.exit(f"Execution stopped. The output file '{output_png}' already exists.")

    df = pd.read_csv(input_csv)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(df['Preprocessed_Review_Text'].values.astype('U'))

    inertias = calculate_inertia(tfidf_matrix, cluster_counts)
    plot_elbow_curve(cluster_counts, inertias, output_png)


if __name__ == '__main__':
    input_csv = './output/word_count_analysis/preprocessed_reviews.csv'
    cluster_counts = range(1, 30)
    output_png = './output/word_count_analysis/png/elbow_analysis.png'
    overwrite = False

    main(input_csv, cluster_counts, output_png, overwrite)
