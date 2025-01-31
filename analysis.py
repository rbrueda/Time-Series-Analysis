import pandas as pd
import nltk
from nltk.corpus import stopwords

#downlopad all available stopwords from nltk to remove from dataframe
nltk.download('stopwords')

stop = stopwords.words('english')

#todo: figure out which posts are more similar than others

#todo: figure out summarizing text models to summarize comments (figure out which overall opinions are the same and which are different)

reddit_df = pd.read_csv('merged_df.csv')

#ensure all rows are strings
reddit_df['title'] = reddit_df['title'].astype(str)
reddit_df['description'] = reddit_df['description'].astype(str)
reddit_df['comments'] = reddit_df['comments'].astype(str)

#add column for reddit posts without stopwords -> need 3: title, description, comments
reddit_df['title_without_stopwords'] = reddit_df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
reddit_df['description_without_stopwords'] = reddit_df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
reddit_df['comments_without_stopwords'] = reddit_df['comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

#print out result in a separate  csv
reddit_df.to_csv('filtered_merged_df.csv', index=False)


#todo: add stemming to data and create word cloud (for analysis visualization) -- also loook into sentiment analysis algorithms and cluster the data based on emotions
#use stemming and  


#extract important columns -> title, description, comments


#idea 2: split all the comments 


#step 1 - exploratory analysis portion -- create a wordcloud to find frequency distribtuion of sample words


# part 1: preprocess data - remove stop words



#part 2: 

#part 1: create a wordcloud to find frequency distribtuion of sample words
# cluster based on categories - then cluster the data based on positive, negative, and neutral connotation


