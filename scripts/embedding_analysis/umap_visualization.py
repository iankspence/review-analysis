import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import umap
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def plot_embeddings_2d(matrix: np.array, sentiments: np.array, output_png: str):
    plt.figure(figsize=(10, 10))
    plt.scatter(matrix[:, 0], matrix[:, 1], c=sentiments, cmap='coolwarm', s=50)
    plt.colorbar()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()


def dimension_reduction(matrix: np.array, method: str, n_components: int):
    if method.lower() == 'umap':
        um = umap.UMAP(n_neighbors=15, n_components=n_components)
        return um.fit_transform(matrix)
    else:
        raise ValueError(f'Unknown method {method}. Please choose "umap".')


def calculate_sentiments(reviews: pd.Series) -> np.array:
    sia = SentimentIntensityAnalyzer()
    return reviews.apply(lambda review: sia.polarity_scores(review)['compound'])


def main(input_csv: str, output_png: str, method: str = 'umap', n_components: int = 2):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    df = pd.read_csv(input_csv)
    df['Embedding'] = df.Embedding.apply(ast.literal_eval).apply(np.array)
    matrix = np.array(df.Embedding.to_list())

    matrix_reduced = dimension_reduction(matrix, method, n_components)

    sentiments = calculate_sentiments(df['Review_Text'])
    plot_embeddings_2d(matrix_reduced, sentiments, output_png)

if __name__ == '__main__':
    input_csv = './output/embedding_analysis/csv/review_embeddings.csv'
    output_png = './output/embedding_analysis/png/embeddings_2d_sentiments.png'  # Output file for the 2D plot
    method = 'umap'  # Using 'umap'
    n_components = 2  # Number of dimensions for UMAP

    main(input_csv, output_png, method, n_components)