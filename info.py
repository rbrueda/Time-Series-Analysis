import pandas as pd

df = pd.read_csv('merged_df.csv')

print(f"# of posts in csv file {len(df)}")