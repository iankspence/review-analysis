import os
import sys
import time

import pandas as pd
import requests
from tqdm import tqdm
from dotenv import load_dotenv

def get_embedding(text: str, api_key: str, model: str = 'text-embedding-ada-002') -> list:
    """
    Sends a request to the OpenAI API to get an embedding for the given text.
    Raises an exception if the API responds with an error.
    """
    url = 'https://api.openai.com/v1/embeddings'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'input': text,
        'model': model,
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['data'][0]['embedding']


def main(input_csv: str, output_csv: str, api_key: str, overwrite: bool = False) -> None:
    """
    Main function to load the raw reviews, get embeddings for each review,
    and save the embeddings to a new CSV file.
    """

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    df_out = pd.read_csv(input_csv)

    # Set up rate limiting
    RATE_LIMIT = 500  # Number of requests per minute
    start_time = time.time()
    request_count = 0

    start_idx = df_out[df_out['Anonymous_Embedding'].notna()].shape[0]
    total_reviews = df_out.shape[0]  # total number of rows

    for idx, review in enumerate(tqdm(df_out['Anonymized_Review_Text'][start_idx:], desc='Processing reviews'), start=start_idx):
        # Check rate limiting
        if request_count >= RATE_LIMIT:
            time_elapsed = time.time() - start_time
            if time_elapsed < 60:
                time.sleep(60 - time_elapsed)
            start_time = time.time()
            request_count = 0

        # Get the embedding for the review
        try:
            embedding = get_embedding(review, api_key)
            df_out.loc[idx, 'Anonymous_Embedding'] = embedding
        except requests.HTTPError as err:
            print(f"Failed to get embedding for review: {review}")
            print(f"Error: {err}")
            df_out.loc[idx, 'Anonymous_Embedding'] = [0] * 1536  # Use a zero vector as a placeholder

        request_count += 1

        # Save progress every 500 reviews or after the last review
        if (idx % 500 == 0) or (idx == total_reviews - 1):
            df_out.to_csv(output_csv, index=False)
            print(f'saving csv at idx: {idx}...')

    # Save the embeddings to a new CSV file
    df_out.to_csv(output_csv, index=False)


if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    input_csv = '/Users/ianspence/Desktop/review-analysis/data/anonymized_reviews_qc6_23743_embeddings.csv'
    output_csv = '/Users/ianspence/Desktop/review-analysis/data/anonymized_reviews_qc6_23743_embeddings.csv'

    overwrite = False

    print(f'input csv: {input_csv}')
    print(f'output csv: {output_csv}')

    main(input_csv, output_csv, api_key, overwrite)