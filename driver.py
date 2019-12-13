import sys
import get_data
from gensimLDA import gensimLDAdriver as runLDA_gs
from make_wordcloud import wordcloud_driver
from run_sklearn import runNMF_sk
from run_sklearn import runLDA_sk
import os.path
from os import path

if __name__ == '__main__':
    search_term = ((input("Enter a topic: ")).lower())
    # Pickle the text
    filename = "./Text/" + search_term.replace(" ", "-") + ".pckl"
    # Check if file exists
    if not path.exists(filename):
        toscrape = int(input("The topic %s does not exist yet. Do you want to scrape it? (1=yes/0=no): " % filename))
        if toscrape == 1:
            print("Getting data now...")
            get_data()
        else:
            print("okee")
            sys.exit()

    algorithm = int(input("Which algorithm would you like to run? (0 = both NMF and LDA, 1 = NMF SciKit, 2 = LDA SciKit, 3 = gensim LDA, 4 = gensim MALLET): "))
    num_topics = 10
    if algorithm == 0:
        runNMF_sk(filename, num_topics)
        runLDA_sk(filename, num_topics)
        wordcloud_driver(search_term, "NMF")
        model_type = "sk-LDA"
    if algorithm == 1:
        runNMF_sk(filename, num_topics)
        model_type = "NMF"
    if algorithm == 2:
        runLDA_sk(filename, num_topics)
        model_type = "sk-LDA"
    if algorithm == 3:
        runLDA_gs(search_term, False)
        model_type = "gen-LDA"
    if algorithm == 4:
        runLDA_gs(search_term, True)
        model_type = "mallet"

    print("visualizing results")
    wordcloud_driver(search_term, model_type)
