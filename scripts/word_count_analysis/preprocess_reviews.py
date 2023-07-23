import os
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(text: str) -> str:
    """
    This function takes a text string as input and performs the following operations:
    - Converts the text to lower case
    - Removes any punctuation
    - Splits the text into individual words
    - Removes any stopwords
    - Lemmatizes the words (shifts it back to its base form)
    - Joins the words back into a single string
    """

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)

    return text


def main(input_csv: str, output_csv: str, overwrite: bool = False) -> None:
    """
    Main function to download necessary NLTK data, load the raw reviews,
    preprocess the 'Review_Text' column, and save the preprocessed data.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    if os.path.exists(output_csv) and not overwrite:
        should_proceed = input(f"The output file '{output_csv}' already exists. Do you want to overwrite it? (y/n): ")
        if should_proceed.lower() != 'y':
            sys.exit(f"Execution stopped. The output file '{output_csv}' already exists.")

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    df = pd.read_csv(input_csv)
    df['Review_Text'] = df['Review_Text'].apply(lambda x: preprocess_text(x))
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    input_csv = './data/cleaned_reviews.csv'
    output_csv = './output/word_count_analysis/csv/preprocessed_reviews.csv'
    overwrite = False

    main(input_csv, output_csv, overwrite)
