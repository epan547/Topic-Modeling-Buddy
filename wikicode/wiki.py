import pickle
import os.path
from os import path
import pandas as pd
import wikipedia

def getpagecontent(userin):
    page_content = ""
    page = wikipedia.page(userin)
    isCorrectPage = int(input("Wikipedia suggested: %s, is this correct? (1:yes/0:no): " % page.title))
    if isCorrectPage:
        page_content = page.content
    return page_content

def wiki_tokens(search_term):
    try:
        wikitxt = getpagecontent(search_term)
        print("Checked wikipedia successfuly")
    except:
        wikitxt = ""
    return wikitxt

if __name__ == "__main__":
    nltk.download('punkt')
    userIn = input("enter a topic: ")
    getpagecontent(userIn)
