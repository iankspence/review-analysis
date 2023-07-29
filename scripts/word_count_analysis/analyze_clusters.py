import os
import sys
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def calculate_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """
    Calculate sentiment polarity of a given text string using Vader.
    """

    # TextBlob
    # return TextBlob(text).sentiment.polarity

    # Vader
    sentiment_scores = analyzer.polarity_scores(text)
    return sentiment_scores['compound']


def calculate_word_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate word counts for each cluster.
    """

    word_count_data = {}

    for cluster in sorted(df['Cluster'].unique()):
        all_reviews_in_cluster = [word for review in df[df['Cluster'] == cluster]['Preprocessed_Review_Text'] for word in review]
        word_counter_in_cluster = Counter(all_reviews_in_cluster)
        most_common_words = [word for word, _ in word_counter_in_cluster.most_common(10)]
        word_count_data[f'Cluster {cluster}'] = pd.Series(word_counter_in_cluster, index=most_common_words)

    word_count_df = pd.DataFrame(word_count_data)
    word_count_df.fillna(0, inplace=True)

    return word_count_df


def plot_word_count_heatmap(df: pd.DataFrame, output_png: str) -> None:
    """
    Plot word count heatmap.
    """

    plt.figure(figsize=(10, 18))
    sns.heatmap(df, cmap='viridis')
    plt.title('Word Count Heatmap')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()


def plot_sentiment_boxplot(df: pd.DataFrame, output_png: str) -> None:
    """
    Plot sentiment boxplot.
    """

    plt.figure(figsize=(10, 8))
    sns.boxplot(x='Cluster', y='Sentiment', data=df)
    plt.title('Sentiment Boxplot per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Sentiment')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()


def main(input_csv: str, heatmap_output_png: str, sentiment_output_png: str, overwrite: bool = False) -> None:
    """
    Main function to load data, preprocess 'Review_Text' column, calculate sentiment,
    count word occurrences, and plot a word count heatmap and a sentiment box plot.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    if os.path.exists(heatmap_output_png) and not overwrite:
        should_proceed = input(
            f"The heatmap output file '{heatmap_output_png}' already exists. Do you want to overwrite it? (y/n): ")
        if should_proceed.lower() != 'y':
            sys.exit(f"Execution stopped. The heatmap output file '{heatmap_output_png}' already exists.")

    if os.path.exists(sentiment_output_png) and not overwrite:
        should_proceed = input(
            f"The sentiment output file '{sentiment_output_png}' already exists. Do you want to overwrite it? (y/n): ")
        if should_proceed.lower() != 'y':
            sys.exit(f"Execution stopped. The output file '{sentiment_output_png}' already exists.")

    df = pd.read_csv(input_csv)

    vader_analyzer = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Preprocessed_Review_Text'].apply(lambda text: calculate_sentiment(text, vader_analyzer))

    word_count_df = calculate_word_counts(df)
    plot_word_count_heatmap(word_count_df, heatmap_output_png)
    plot_sentiment_boxplot(df, sentiment_output_png)


if __name__ == '__main__':

    input_csv = './output/embedding_analysis/csv/preprocessed_review_embeddings_with_20_clusters_50_pca.csv'
    heatmap_output_png = './output/embedding_analysis/png/word_count_heatmap_preprocessed_review_embeddings_with_20_clusters_50_pca.png'
    sentiment_output_png = './output/embedding_analysis/png/sentiment_boxplot_preprocessed_review_embeddings_with_20_clusters_50_pca.png'
    overwrite = False

    main(input_csv, heatmap_output_png, sentiment_output_png, overwrite)
