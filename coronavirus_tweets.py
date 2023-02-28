# Part 3: Mining text data.
import pandas as pd
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import urllib
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'coronavirus_tweets.csv'.

def read_csv_3(data_file):
    df = pd.read_csv(data_file, encoding='latin-1')
    return df


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    return df['Sentiment'].unique()


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    return df['Sentiment'].value_counts()[1:2].index.tolist()[0]


# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    a = df.loc[df['Sentiment'] == 'Extremely Negative']
    b = a['TweetAt'].value_counts()[:1].index.tolist()
    return b[0]


# Modify the dataframe df by converting all tweets to lower case.
def lower_case(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-zA-Z]', ' ')


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    df['OriginalTweet'] = df['OriginalTweet'].replace('\s+', ' ', regex=True)


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df['OriginalTweet'] = df['OriginalTweet'].apply(word_tokenize)


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    return tdf['OriginalTweet'].explode().value_counts().sum()


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    return tdf['OriginalTweet'].explode().drop_duplicates().value_counts().sum()


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
    return tdf['OriginalTweet'].explode().value_counts()[0:k]


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt

def remove_stop_words(tdf):
    link = "https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt"
    f = urllib.request.urlopen(link)
    myfile = f.read()
    stop_words = [str(i)[2:-1] for i in myfile.split()]
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(
        lambda x: [item for item in x if item not in stop_words and len(item) > 2])


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.

def stemming(tdf):
    ps = PorterStemmer()
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [ps.stem(i) for i in x if i != ''])


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    tweets = df.loc[:, "OriginalTweet"].astype(str).to_numpy()
    sentiments = df.loc[:, 'Sentiment'].to_numpy()

    vec = CountVectorizer(analyzer="word", ngram_range=(2, 6), stop_words='english')
    x_train = vec.fit_transform(tweets)
    mnb = MultinomialNB()
    mnb.fit(x_train, sentiments)
    y_pre = mnb.predict(x_train)
    return y_pre


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive')
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
    score = accuracy_score(y_pred, y_true)
    return '%.3f' % score
