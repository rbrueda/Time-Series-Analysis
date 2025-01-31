
#todo: merge data and cluster based off text similarity (use sbert for)
import pandas as pd
import os

df_list = []

#get all the combined reddit data and but in into 1 dataframe

folder_name = 'Reddit-Data'

folder_path = os.path.abspath(folder_name)

for root, dirs, files in os.walk(folder_path):
    for file in files:
        df = pd.read_csv(root + '/' + file)
        df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

#save the results in a csv
merged_df.to_csv("merged_df.csv", index=False)

