import csv
import nltk
import json
import time
from datetime import datetime
import tweepy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.linear_model import LogisticRegression


# ___ START OF DATA STORAGE CLASS AND ASSOCIATED FUNCTIONS___

# Stores data in the form of a binary sentiment value and review.
class DataStore:
    def __init__(self, sentiment, review):
        self.sentiment = sentiment
        self.review = review

    def __repr__(self):
        return f'Sentiment: {self.sentiment}, Review: {self.review}'


# Makes instances of the Datastore class based on data from a CSV or TSV file. File must be contain at least
# 'sentiment' and 'review' elements to be compatible
def sentiment_review_storage(filename='labeledTrainData.tsv'):
    file_import = csv.DictReader(open(filename, encoding='utf-8'), delimiter='\t')
    class_list = []

    for dictionary in file_import:
        s = int(dictionary.get('sentiment'))
        r = (dictionary.get('review').lower())
        class_list.append(DataStore(s, r))

    return class_list


# Pulls data from selected hashtag and stores it in DataStore with a sentiment value of None
def get_tweets(hashtag=input("Enter hashtag to be scraped. Must start with #: ")):
    # Keys and secrets redacted
    tweet_class_list = []

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth)

    search_target = hashtag + ' -filter:retweets'
    # date_since = datetime.today().strftime('%Y-%m-%d')

    tweets = tweepy.Cursor(api.search,
                           q=search_target,
                           lang='en',
                           result_type='recent').items(1000)

    for tweet in tweets:
        tweet_class_list.append(DataStore(None, tweet.text))

    return tweet_class_list


# ___START OF SENTIMENT ANALYSIS FUNCTIONS___

# Takes the reviews from the sentiment_review_storage function and tokenizes the review portion.
def tokenize(untokenized_data=None):
    # nltk.download('punkt')

    if untokenized_data is None:
        untokenized_data = sentiment_review_storage()

    for item in untokenized_data:
        item.review = word_tokenize(item.review)

    return untokenized_data


# Removes stopwords from tokenized data
def remove_stopwords(tokenized_data=None):
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    if tokenized_data is None:
        tokenized_data = tokenize()

    for item in tokenized_data:
        for word in item.review:
            if word in stop_words:
                item.review.remove(word)

    return tokenized_data


# Takes tokenized strings and an integer dictating the number of feature words to select, and selects feature words
# based on the overall sentiment value of each word.
def features(tokenized_training_data=None, feature_word_number=1000, feature_word_occurrence_rate=250):
    word_dict = {}
    occur_dict = {}
    feature_dict = {}

    if tokenized_training_data is None:
        tokenized_training_data = remove_stopwords()

    # Makes dictionaries containing every word in the tokenized review along with their base sentiment scores
    # and occurrence rates
    for item in tokenized_training_data:
        for word in item.review:
            if word in word_dict:
                if item.sentiment == 1:
                    word_dict[word] += 1
                else:
                    word_dict[word] -= 1
            else:
                if item.sentiment == 1:
                    word_dict[word] = 1
                else:
                    word_dict[word] = -1
            if word in occur_dict:
                occur_dict[word] += 1
            else:
                occur_dict[word] = 1

    # Filters out words not meeting the occurrence rate criteria
    keys = [k for k, v in occur_dict.items() if v < feature_word_occurrence_rate]
    for x in keys:
        del word_dict[x]
        del occur_dict[x]

    # Divides word sentiment by occurrence rate to get sentiment score
    for word in word_dict:
        word_dict[word] = int(word_dict[word]) / int(occur_dict[word])

    # Gets the most highest and lowest scored words based on the number of feature words specified in the
    # form of a tuple

    # Can I find a way to append highest value key, value pair from one dict to another in
    # a loop? could make this easier
    pos_neg_amount = round(feature_word_number / 2)
    dict_counter = Counter(word_dict)
    high = dict_counter.most_common(pos_neg_amount)
    low = dict_counter.most_common()[-pos_neg_amount - 1:-1]
    total = high + low

    # Converts the Tuples back to dictionaries
    tup_con(total, feature_dict)

    # Converts the feature dictionary to a list, then reconverts to a dictionary
    # with its index in the list as its value
    keylist = list(feature_dict.keys())
    # noinspection PyTypeChecker
    return_feature_dict = dict(map(reversed, enumerate(keylist)))

    dict_con(return_feature_dict, 'feature_words')
    return return_feature_dict


# Takes tokenized data and creates bitvectors/ sentiment tuples using the feature word dictionary. tokenized words are
# assigned a 1 or 0 depending on whether or not the word is in the feature list
def create_feature_vector(dictionary=None, tokens=None):
    feature_vector_list = []

    if dictionary is None:
        dictionary = features()
    if tokens is None:
        tokens = tokenize()

    for item in tokens:
        sentimentval = item.sentiment
        tup = (bitvec_con(item.review, dictionary), sentimentval)
        feature_vector_list.append(tup)

    return feature_vector_list


# Creates a bitvector using the tweet data. No sentiment included
def create_tweet_vector(dictionary=None, tokens=None):
    tweet_vector_list = []

    if dictionary is None:
        dictionary = features()
    if tokens is None:
        tokens = remove_stopwords(tokenize(get_tweets()))

    for item in tokens:
        tup = bitvec_con(item.review, dictionary)
        tweet_vector_list.append(tup)

    return tweet_vector_list


# Takes a tuple of format (review, sentiment) and trains using logistic regression. Proceeds to do logistic regression
# on the set of tweet vectors
def train_sentiment(rev_sent_tup=None, tweet_data=None):
    logreg = LogisticRegression()
    x = []
    y = []

    if rev_sent_tup is None:
        rev_sent_tup = create_feature_vector()
    if tweet_data is None:
        tweet_data = create_tweet_vector()

    for item in rev_sent_tup:
        x.append(item[0])
        y.append(item[1])

    logreg.fit(x, y)

    predict = logreg.predict(tweet_data)
    print(predict)

    return predict


# ___START OF "HELPER" FUNCTIONS___

# Converts tuples to dictionaries
def tup_con(tup, dictionary):
    for a, b in tup:
        dictionary.setdefault(a, []).append(b)

    return dictionary


# Converts a string of tokenized text into bitvector
def bitvec_con(data, dictionary):
    output = []
    text = data
    for item in dictionary:
        if item in text:
            output.append(1)
        else:
            output.append(0)

    return output


# Converts a dictionary to a JSON file
def dict_con(dictionary, name):
    with open(name, 'w') as outfile:
        json.dump(dictionary, outfile)


if __name__ == '__main__':
    train_sentiment()
