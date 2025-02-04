import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
import string
import spacy
import re
import json
import heapq

#for visualizing data
import matplotlib.pyplot as plt

# Ensure required resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

reddit_df = pd.read_csv('merged_df.csv')

#ensure all rows are strings
reddit_df['title'] = reddit_df['title'].astype(str)
reddit_df['description'] = reddit_df['description'].astype(str)
reddit_df['comments'] = reddit_df['comments'].astype(str)

#print out result in a separate csv
reddit_df.to_csv('filtered_merged_df.csv', index=False)

#remove multiple spaces and strip words
clean_title_text = reddit_df['title'].astype(str).str.replace(r"\n", " ", regex=True).replace(r"\s+", " ", regex=True)
clean_title_text = clean_title_text.astype(str).str.strip()

clean_description_text = reddit_df['title'].astype(str).str.replace(r"\n", " ", regex=True).replace(r"\s+", " ", regex=True)
clean_description_text  = clean_title_text.astype(str).str.strip()

clean_comments_text = reddit_df['title'].astype(str).str.replace(r"\n", " ", regex=True).replace(r"\s+", " ", regex=True).replace(";;;", " ")
clean_comments_text = clean_comments_text.astype(str).str.strip()

#merge clean_title_text, clean_description_text, clean_comments_text
clean_text = clean_title_text.str.cat(
    [clean_description_text, clean_comments_text], sep=" ", na_rep=""
)

#tokenize the cleaned text
tokens = clean_text.astype(str).apply(word_tokenize)

#only extract words -> alphabetical text
filtered_tokens_alpha = tokens.apply(lambda words: [word for word in words if word.isalpha()])
filtered_tokens_alpha = filtered_tokens_alpha.explode()  # Converts list elements into separate rows

#remove capitalization in dataframe
filtered_tokens_alpha = filtered_tokens_alpha.apply(lambda x: x.lower() if isinstance(x, str) else x)

# Remove stop words 
filtered_tokens_final = filtered_tokens_alpha[~filtered_tokens_alpha.isin(stop_words)]

filtered_tokens_final.to_csv("tokens-without-stopwords.csv", index=False)

#todo: figure out issue with "the" not getting extracted!
#todo: print out result here

#put tokens onto stemming algorithm, Porter Stemmer
p_stemmer = PorterStemmer()

stemmed_tokens = [p_stemmer.stem(word) for word in filtered_tokens_final]

# print(f"# of stemmed tokens found: {len(set(stemmed_tokens))}") #1682 unique tokens found in dataset
# print(f"Number of stemmed words (non unique): {len(stemmed_tokens)}") #25295 non unique tokens found

freq_distribution = {}

#write frequency distribution of stemmed words 
#we have to go through all stemmed words and put to hashmap (write result in external file)
for token in stemmed_tokens:
    if token not in freq_distribution:
        freq_distribution[token] = 1

    else:
        freq_distribution[token] += 1

#write results in external file
with open('freq-distribution.txt', 'w') as file:
    file.write(json.dumps(freq_distribution))

#generate top 20 most popular words and plot statistics onto graph
largest_items = heapq.nlargest(20, freq_distribution.items(), key=lambda x: x[1])

#extract the keys and values with this property
top_dict = dict(largest_items)

#find freqnency distribution of stemmed words
words = [word for word in top_dict]
count = [count for count in top_dict.values()]

print(words)
print(count)

plt.bar(words, count)
plt.title('Frequency Count fo 20 Popular Words')
plt.xlabel('Words')
plt.ylabel('Count')
plt.show()



#future? -> figure out with supervisor -> categorize data (use the stemming data analysis?)

#todo: figure out sentiment scoring here -- find posts with overall positive and negative connation (in general and then show how the sentiments change over time) -- cateogize by dates, and then check sentiment scores per post (meaning strength of negative, neutral, and positive results)

#todo: add frequency distribution

#todo; perform clustering on the data





#extract important columns -> title, description, comments


#idea 2: split all the comments 


#step 1 - exploratory analysis portion -- create a wordcloud to find frequency distribtuion of sample words


# part 1: preprocess data - remove stop words



#part 2: 

#part 1: create a wordcloud to find frequency distribtuion of sample words
# cluster based on categories - then cluster the data based on positive, negative, and neutral connotation


