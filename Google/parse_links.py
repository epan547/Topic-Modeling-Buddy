import pickle
from bs4 import BeautifulSoup
import requests
import string
import nltk
import re
import os.path
from os import path
from Google.google_scrape import scrape_google
from Google.medium_scrape import scrape_google as scrape_medium

def get_links(search_term):
    # Unpickle the list of links
    lst = []
    file_name = "./Google/Links/" + search_term 
    overWrite = int(input("Do you want to scrape Google? (1=yes/0=no): "))
    if overWrite == 1:
        scrape_medium(search_term)
    fileobject = open(file_name, 'rb')
    lst = pickle.load(fileobject)
    return lst

# Function for cleaning text of punctuation, symbols, and 1-letter words
def clean_content(article):
    # substitute in a regular apostrophe for '’' to word with word_tokenize
    article = article.replace('’', "").replace("'", "").replace(".", "")
    article = re.sub(r'[^a-zA-Z]', ' ', article)
    article = article.strip(string.punctuation).lower()
    tokens = nltk.tokenize.word_tokenize(article)
    words = list(filter(lambda w: any(x.isalpha() for x in w), tokens))
    for index, token in enumerate(words):
        if len(token) <= 3:
            del words[index]
    return article


# Scrape the text from those links
def scrape_text(link):
    try:
        page = requests.get(link)
    except:
        print("Wasn't able to get: ", link)
        return ""
    soup = BeautifulSoup(page.content, 'html.parser')
    html_txt = soup.find_all('p')
    all_txt = ""
    for p in html_txt:
        all_txt += (p.get_text()) + ' '
    clean_txt = clean_content(all_txt)
    return clean_txt

def pickle_links(links, search_term):
    # Pickle the list
    filename = "./Google/Links/" + search_term
    # open the file for writing
    picklefile = open(filename,'wb')
    # this writes the object a to the
    pickle.dump(links, picklefile)
    # here we close the fileObject
    picklefile.close()


# Generate text file of text from all links
def parse_links(search_term):
    lst = get_links(search_term)
    links = []
    text = []
    i = 1
    for link in lst:
        t = scrape_text(link)
        text.append(t)
        if len(text) > len(links):
            links.append(link)
        print("Articles done: ", i)
        i += 1
    pickle_links(links, search_term)
    return text, links;
