from bs4 import BeautifulSoup as bs
import requests
import pickle
import os.path
from os import path

def getWikiLinks():
    userin = (input("Enter a topic: ")).lower()).replace(" ", "-")
    filename = "./topics/" + userin + "-links.pckl"

    if path.exists(filename):
        infile = open(filename, "rb")
        topic = pickle.load(infile)
        print(type(topic))
        for line in topic.keys():
            print(topic[line])
        infile.close()

    else:
        pcklfile = open(filename, "wb")
        res = requests.get(input("enter a url: "))
        soup = bs(res.text, "html.parser")
        articles = {}
        soup = soup.find(id="bodyContent")
        for link in soup.find_all("a"):
            url = link.get("href", "")
            if "/wiki/" in url:
                articles[link.text.strip()] = url
        for link in articles.keys():
            articles[link] = 'https://en.wikipedia.org' + articles[link]
            print(articles[link])
        pickle.dump(articles, pcklfile)
        pcklfile.close()
