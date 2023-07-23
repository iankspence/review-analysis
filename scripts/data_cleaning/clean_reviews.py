import pandas as pd
import os

def clean_reviews(input_dir, output_csv):
    # Initialize a list to store all dataframes
    df_list = []

    # Walk through all files in the directory and its subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                # Full path to the csv file
                csv_path = os.path.join(root, file)

                # Load data from the csv file
                df = pd.read_csv(csv_path)

                # Map old column names to new ones
                column_mapping = {
                    'd4r55': 'Author_Name',
                    'RfnDt': 'Author_Review_Count',
                    'rsqaWe': 'Review_Date',
                    'wiI7pd': 'Review_Text',
                    'DZSIDd': 'Response_Date',
                    'wiI7pd 2': 'Response_Text'
                }

                # Rename the columns
                df.rename(columns=column_mapping, inplace=True)

                # Ensure 'Author_Name' is a string and not empty
                if 'Author_Name' in df.columns:
                    df['Author_Name'] = df['Author_Name'].astype(str)
                    df = df[df['Author_Name'].str.strip() != ('' or 'nan')]

                # Ensure 'Review_Text' is a string
                if 'Review_Text' in df.columns:
                    df['Review_Text'] = df['Review_Text'].astype(str)
                    # Filter out the rows where 'Review_Text' ends with '…'
                    df = df[~df['Review_Text'].str.endswith('…')]
                    # Filter out the rows where 'Review_Text' is 'nan'
                    df = df[df['Review_Text'] != 'nan']

                # Ensure 'Response_Text' is a string
                if 'Response_Text' in df.columns:
                    df['Response_Text'] = df['Response_Text'].astype(str)

                # Create a new DataFrame with only the desired columns
                columns_to_keep = [
                    'Author_Name',
                    'Author_Review_Count',
                    'Review_Date',
                    'Review_Text',
                    'Response_Date',
                    'Response_Text'
                ]
                df = df[df.columns.intersection(columns_to_keep)]

                # Add 'State_Province' and 'Clinic_Name' columns
                df['State_Province'] = os.path.basename(root)
                df['Clinic_Name'] = os.path.splitext(file)[0]

                # Append the cleaned dataframe to the list
                df_list.append(df)

    # Concatenate all dataframes in the list
    all_data = pd.concat(df_list, ignore_index=True)

    # Save all cleaned data to a new csv file
    all_data.to_csv(output_csv, index=False)


if __name__ == '__main__':
    input_dir = '../../data/raw_data'
    output_csv = '../data/cleaned_reviews.csv'

    clean_reviews(input_dir, output_csv)