
#for sentiment analysis 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#for data manipulation
import pandas as pd

df = pd.read_csv("merged_df.csv")

#stage 1: figure out overall sentiment trends through entirety of data (ordered by posted_date)

#sentiment analysis using VADER - python
#VADER - Valence Aware Dictionary and Sentiment Reasoner
#can be displayed with the following components -> positive, negative, neutral, compound
def sentiment_scores(comment: str):
    sid_obj = SentimentIntensityAnalyzer()

    #polarity_socores method of SentimentIntensityAnalyzer object gives a sentiment dictionary
    #contains pos, neg, neu, and compound scores
    sentiment_dict = sid_obj.polarity_scores(comment)

    #print out results
    print(f"Comment was rated {sentiment_dict['neg']*100}% Negative")
    print(f"Comment was rated {sentiment_dict['neu']*100}% Neutral")
    print(f"Comment was rated {sentiment_dict['pos']*100}% Positive")

    return sentiment_dict


#go through each post in merged_df.csv and the compute sentiment scores 
#compute average of neutral, positive, and negative results
for row in df.iterrows():
    average_pos_score = 0
    average_neg_score = 0
    average_neutral_score = 0

    current_count = 0
    #go through title, description, and title for posts
    
    #treat description and title as separate values
    #todo: see if title should be determined in overall scoring or not
    res1 = sentiment_scores(row['title'])
    average_pos_score += res1['pos']
    average_neg_score += res1['neg']
    average_neutral_score += res1['neu']
    current_count += 1

    #check if description is empty or not
    if row['description'] != "":
        res2 = sentiment_scores(row['description'])
        average_pos_score += res2['pos']
        average_neg_score += res2['neg']
        average_neutral_score += res2['neu']
        current_count += 1


    #iterate through each comment in comment
    for comment in row['comments'].split(';;;'):
        res3 = sentiment_scores(row['comments'])
        average_pos_score += res3['pos']
        average_neg_score += res3['neg']
        average_neutral_score += res3['neu']

        current_count += 1

    average_pos_score = average_pos_score / current_count
    average_neg_score = average_neg_score / current_count
    average_neutral_score = average_neutral_score / current_count

    #write the results in an external file


#todo: run code and see if it works
#todo: write results in external csv file
#todo: sort data based on posted date
#todo: maybe -- average dates a well (if there are a lot of repetive values)
#todo: display data in line graph (3 for pos, neg, and neutral) for sentiment change -- dont do categorical for now






    #go through each comment in merged_df.csv
