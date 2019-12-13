# Start with loading all necessary libraries
import numpy as np
import pickle
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import math
import matplotlib.pyplot as plt

from run_sklearn import runNMF_sk, get_weights, runLDA_sk

"""model_type:
"NMF": sklearn NMF
"sk-LDA": sklearn LDA
"gen-LDA": gensim
"mallet": gensim mallet
"""

def create_input(weight_lst):
    input = ""
    for (weight, word) in weight_lst:
        input += (word + " ")*int(round(weight,2)*100)
    return input

def load_topics(search_term, model_type):
    if model_type == "mallet":
        filename = "./Topics/" + search_term.replace(" ", "-") + "-mallet.pckl"
    elif model_type == "gen-LDA":
        filename = "./Topics/" + search_term.replace(" ", "-") + ".pckl"
    f = open(filename, "rb")
    topics = pickle.load(f)
    f.close()
    return topics

def parse_topics(topics):
    """FOR GENSIM. Creates string to use as input for wordcloud
    based on output of gensimLDA.py

    Returns:
    -num_topics: the number of topics
    -wc_in: list of input strings for the wordcloud
    """
    wc_in = []
    keywords = [topic[1] for topic in topics]
    num_topics = len(topics)
    for word in keywords:
        k = ""
        #create list of strings, len(keywords) in topic
        weight_word_str = word.split(" + ")

        #for each string, separate into weight and keyword
        for term in weight_word_str:
            weight, key = term.split("*")

            #scale weight, add appropriate num to input string
            w = int(float(weight)*1000)
            k += (key.strip('"') + " ") * w
        wc_in.append(k)
    return num_topics, wc_in

def pick_suffix(model_type):
    suffix = ""
    if model_type == "NMF":
        suffix = "-nmfsklearn"
    elif model_type == "sk-LDA":
        suffix = "-ldasklearn"
    elif model_type == "gen-LDA":
        suffix = "-gensim"
    elif model_type == "mallet":
        suffix = "-mallet"
    return suffix

def plot_wordcloud(search_term, wordclouds, num_topics, model_type):
    print(num_topics)
    if num_topics >= 4:
        ncols = 4
        nrows = math.ceil(num_topics/ncols)
    else:
        ncols = 1
        nrows = num_topics
    fig, axs = plt.subplots(nrows, ncols)
    fig.suptitle('Search term: ' + search_term)

    for ax, i in zip(axs.flatten(), range(nrows*ncols)):
        if i >= num_topics:
            fig.delaxes(ax)
        else:
            ax.imshow(wordclouds[i], interpolation='bilinear')
            ax.axis("off")

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    suffix = pick_suffix(model_type)
    plt.savefig("./Wordclouds/" + search_term.replace(" ", "-") + suffix + '.png')

def wc_sklearn(search_term, model_type):
    num_topics = 12
    filename = "./Text/" + search_term.replace(" ", "-") + ".pckl"

    if model_type == "NMF":
        (model, vectorizer) = runNMF_sk(filename, num_topics)
    elif model_type== "sk-LDA":
        (model, vectorizer) = runLDA_sk(filename, num_topics)
    else:
        print("Invalid model")
        return

    weights = get_weights(model, vectorizer)

    wordclouds = []
    for i in range(num_topics):
        # Start with one review:
        text = create_input(weights[i])
        # Create and generate a word cloud image:
        wordclouds.append(WordCloud(background_color='white', collocations=False).generate(text))

    # Display the generated image:
    plot_wordcloud(search_term, wordclouds, num_topics, model_type)


def wc_gensim(search_term, model_type):
    #load in topics, format for wordcloud
    topics = load_topics(search_term, model_type)
    num_topics, texts = parse_topics(topics)

    wordclouds = []
    for text in texts:
        # Create and generate a word cloud image:
        wordclouds.append(WordCloud(collocations=False).generate(text))

    # Display the generated image:
    plot_wordcloud(search_term, wordclouds, num_topics, model_type)

def wordcloud_driver(search_term, model_type):
    # search_term = ((input("Enter a processed topic: ")).lower())
    # model_type = int(input("What kind of model are you using?: 1:NMF sklearn, 2:LDA sklearn, 3:gensim, 4:mallet "))
    if model_type == "NMF" or model_type=="sk-LDA":
        wc_sklearn(search_term, model_type)
    elif model_type == "gen-LDA" or model_type == "mallet":
        wc_gensim(search_term, model_type)
