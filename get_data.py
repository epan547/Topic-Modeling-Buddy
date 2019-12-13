from Google.parse_links import parse_links, get_links
from wikicode.wiki import *
import os.path
from os import path
import nltk
import string
import sys
import pickle
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction import text


def clean_content(article):
    clean_words = []
    # Initialize nltk lemmatizer
    lemmatizer = WordNetLemmatizer()
    #remove ', `, and .
    article = article.replace('’', "").replace("'", "").replace(".", "")
    #remove additional punctuation and convert to lowercase
    article = article.strip(string.punctuation).lower()
    tokens = nltk.tokenize.word_tokenize(article)
    words = list(filter(lambda w: any(x.isalpha() for x in w), tokens))
    stopwords = text.ENGLISH_STOP_WORDS.union(["yarn", "html", "bitly", "http", "loading", "working", "twittercom", "wwwyoutubecom", "https", "facebookcom", "wwwfacebookcom"])
    for index, token in enumerate(words):
        lemma = lemmatizer.lemmatize(token)
        if len(lemma)>3 and lemma not in stopwords:
            clean_words.append(lemma)
    return clean_words

def create_df(clean_tokens, links):
    data = []
    print(links)
    for lst in clean_tokens:
        data.append((' ').join(lst))
    print("\nlength of clean tokens: ", len(clean_tokens))
    print("\nlength of links: ", len(links))
    # df = pd.DataFrame({'data': data, 'links': links})
    df = pd.DataFrame({'data': data})
    return df

def pickle_gensim(search_term, words):
    filename = "./Text/" + search_term.replace(" ", "-") + "words.pckl"
    f = open(filename, 'wb')
    pickle.dump(words, f)
    f.close()

def pickle_sk(filename, words):
    f = open(filename, 'wb')
    pickle.dump(words, f)
    f.close()

def pickle_links(search_term, links):
    filename = "./Text/" + search_term.replace(" ", "-") + "links.pckl"
    f = open(filename, 'wb')
    pickle.dump(links, f)
    f.close()

def tokenize_documents(mediumtxt, wikitxt):
    tokens = []
    for doc in mediumtxt:
        clean_tokens = clean_content(doc)
        tokens.append(clean_tokens)
    tokens.append(clean_content(wikitxt))
    return tokens

def pickle_df(filename, df):
    f = open(filename, 'wb')
    pickle.dump(df, f)
    f.close()

if __name__ == '__main__':
    search_term = ((input("Enter a topic: ")).lower())
    # Pickle the text
    filename = "./Text/" + search_term.replace(" ", "-") + ".pckl"
    # Check if file exists
    if path.exists(filename):
        overWrite = int(input("This topic's content has been pickled at %s, do you want to overwrite? (1=yes/0=no): " % filename))
        if overWrite == 0:
            print("okee")
            sys.exit(0)
            # Exit program

    # Scrape from Wikipedia
    wikitxt = wiki_tokens(search_term)
    # Scrape from Google
    medium_articles, links = parse_links(search_term)
    links.append('wikipedia')
    # Clean tokens and save in a pickle file
    clean_tokens = tokenize_documents(medium_articles, wikitxt)
    print(len(clean_tokens))
    pickle_cleaned_text(search_term, clean_tokens)
    # Create a dataframe and save in a pickle file
    df = create_df(clean_tokens, links)
    pickle_df(filename, df)
