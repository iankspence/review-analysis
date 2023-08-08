import os
import ast
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


def load_env_vars() -> dict:
    """
    Load database environment variables.
    """
    load_dotenv()

    return {
        'username': os.getenv('DB_USERNAME'),
        'password': os.getenv('DB_PASSWORD'),
        'db_name': os.getenv('DB_NAME'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }


def create_sqlalchemy_engine(credentials: dict):
    """
    Create a SQLAlchemy engine using database credentials.
    """
    return create_engine(
        f"postgresql://{credentials['username']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['db_name']}")


def load_data(input_csv: str) -> pd.DataFrame:
    """
    Load data from the CSV file and prepare it for insertion into the database.
    """
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['Anonymous_Embedding'])

    df['Anonymous_Embedding'] = df['Anonymous_Embedding'].apply(ast.literal_eval)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    return df


def insert_data(df: pd.DataFrame, engine, table_name: str = 'reviews') -> None:
    """
    Insert data from DataFrame into PostgreSQL.
    """
    df.to_sql(table_name, engine, if_exists='append', index=False)


def main(input_csv: str, overwrite: bool = False) -> None:
    """
    Main function to load data from a CSV file and insert it into a PostgreSQL database.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"The input file '{input_csv}' does not exist.")

    credentials = load_env_vars()
    engine = create_sqlalchemy_engine(credentials)
    df = load_data(input_csv)

    insert_data(df, engine)


if __name__ == '__main__':
    input_csv = '/Users/ianspence/Desktop/review-analysis/data/anonymized_reviews_qc6_23743_embeddings.csv'
    overwrite = False

    main(input_csv, overwrite)
