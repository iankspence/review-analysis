import pandas as pd
import os


def clean_reviews(input_dir, output_csv):

    df_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):

                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)

                column_mapping = {
                    'd4r55': 'Author_Name',
                    'RfnDt': 'Author_Review_Count',
                    'rsqaWe': 'Review_Date',
                    'wiI7pd': 'Review_Text',
                    'DZSIDd': 'Response_Date',
                    'wiI7pd 2': 'Response_Text'
                }

                df.rename(columns=column_mapping, inplace=True)

                if 'Author_Name' in df.columns:
                    df['Author_Name'] = df['Author_Name'].astype(str)
                    df = df[df['Author_Name'].str.strip() != ('' or 'nan')]

                if 'Review_Text' in df.columns:
                    df['Review_Text'] = df['Review_Text'].astype(str)
                    df = df[~df['Review_Text'].str.endswith('…')]
                    df = df[df['Review_Text'] != 'nan']

                if 'Response_Text' in df.columns:
                    df['Response_Text'] = df['Response_Text'].astype(str)

                columns_to_keep = [
                    'Author_Name',
                    'Author_Review_Count',
                    'Review_Date',
                    'Review_Text',
                    'Response_Date',
                    'Response_Text'
                ]
                df = df[df.columns.intersection(columns_to_keep)]

                df['State_Province'] = os.path.basename(root)
                df['Clinic_Name'] = os.path.splitext(file)[0]

                df_list.append(df)

    all_data = pd.concat(df_list, ignore_index=True)
    all_data.to_csv(output_csv, index=False)


if __name__ == '__main__':
    input_dir = '../../data/raw_data'
    output_csv = '../data/cleaned_reviews.csv'

    clean_reviews(input_dir, output_csv)