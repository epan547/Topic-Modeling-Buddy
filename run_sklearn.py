from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import os.path
from os import path
from datetime import date
import time
import numpy as np


def loadDataFrame(filename):
    if path.exists(filename):
        infile = open(filename, "rb")
        df = pickle.load(infile)
        infile.close()
        return df
    else:
        return ("This topic has not yet been pickled. Run get_data.py first.")

def get_links(keyword):
    new_key = keyword.replace("-"," ")
    p = "./Google/Links/"+ new_key
    print(p)
    if path.exists(p):
        infile = open(p, "rb")
        links = pickle.load(infile)
        infile.close()
        return links
    else:
        return ("Cannot find pickle file of links :(")


def get_weights(model, count_vectorizer, n_top_words=10):
    """
    weights = a dictionary where each key is a topic, and each value is a list of tuples (weight, word).
    """
    words = count_vectorizer.get_feature_names()
    weights = {}
    for index, topic in enumerate(model.components_):
        weights[index] = []
        for i in topic.argsort()[:-n_top_words -1:-1]:
            weights[index].append((topic[i], words[i]))
    # print(weights)
    return weights


def print_topics(model, count_vectorizer, relevant_links, n_top_words=5):
    to_save = []
    words = count_vectorizer.get_feature_names()

    for index, topic in enumerate(model.components_):
        to_save.append("\nTopic: "+str(index)+"\n")
        print("\nTopic: ", index)
        topics = []
        for i in topic.argsort()[:-n_top_words -1:-1]:
            topics.append(words[i])
        topicstr = " ".join(topics)
        print(topicstr)
        to_save.append(topicstr)
        if index in relevant_links:
            to_save.append("\nMost relevant links: ")
            for link in relevant_links[index]:
                print(link)
                to_save.append(link+"\n")
    return to_save


def get_doc_topics(alg_object, count_data, links):
    doc_topic = alg_object.transform(count_data)
    print("doc topic shape: ", doc_topic.shape)
    relevant_articles = {}
    for n in range(doc_topic.shape[0]):
        maintopic = doc_topic[n].argmax()
        value = doc_topic[n][maintopic]
        if maintopic in relevant_articles:
            if n < len(links):
                relevant_articles[maintopic][0].append(value)
                relevant_articles[maintopic][1].append(links[n])
        else:
            if n < len(links):
                relevant_articles[maintopic] = [[value],[links[n]]]
        # if n < len(links):
            # print("doc: %d, topic: %s, \n value: %f, \nlink: %s\n" % (n, maintopic, value, links[n]))
        # else:
            # print("doc: %d, topic: %s, \n value: %f \n" % (n, maintopic, value))
    return relevant_articles

def get_relevant_links(relevant_articles, n_links):
    """
    n_links = number of links to return for each topic
    relevant_articles = dictionary data structure with topic number as key,
    and a list of [[topic relevance scores], [article links]] as the value
    """
    relevant_links = {}
    for topic in relevant_articles.keys():
        relevant_links[topic] = []
        for i in range(n_links):
            value_lst = (relevant_articles[topic][0])
            link_lst = relevant_articles[topic][1]
            if i+1 > len(value_lst):
                break
            # print("Value list length: ", len(value_lst), " ", i)
            # Getting the index of the maximum value and removing it so it's not duplicated in next iteration
            max_index = value_lst.index(max(value_lst))
            del value_lst[max_index]
            # Removing and getting the link at the max value index
            max_value_link = link_lst.pop(max_index)
            # Adding that link to the list of most relevant links for this topic
            relevant_links[topic].append(max_value_link)
    return relevant_links


def runLDA_sk(filename, number_topics=10):
    # unpickle dataframe
    df = loadDataFrame(filename)

    # unpickle links
    keyword = filename[7:(len(filename)-5)]
    links = get_links(keyword)
    links.append('Wikipedia')
    print("Length of links: ", len(links))

    # Vectorize data from df -> Bag of Words
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    count_data = count_vectorizer.fit_transform(df['data'])

    # Initialize data being saved in text file and adding the date
    to_save = []
    to_save.append(str(date.today()))

    # Run and time LDA, since it usually takes a while to converge
    print("\nRunning LDA with SciKit Learn...\n")
    start = time.time()
    lda = LDA(n_components=11, max_iter=1000, learning_offset=50.,random_state=10).fit(count_data)
    end = time.time()
    print('Runtime:', end - start)

    # Print topics per document
    relevant_articles = get_doc_topics(lda, count_data, links)
    relevant_links = get_relevant_links(relevant_articles, 2) # TODO: Change this to user input

    # Save the result of LDA in .txt file
    to_save.extend(print_topics(lda, count_vectorizer, relevant_links))
    save_topics("./LDA-SK_Results/"+keyword+".txt", to_save)

    # Log Likelihood: Higher the better
    print("\nLog Likelihood (Higher=better): ", lda.score(count_data))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity (Lower=better): ", lda.perplexity(count_data))

    return lda, count_vectorizer;

    # Code from Paul
    # normalized_by_column = lda.components_ / lda.components_.sum(axis=0)
    # print(normalized_by_column[0].argmax())
    # print(count_vectorizer.get_feature_names(normalized_by_column[0].argmax()))



def runNMF_sk(filename, num_topics):
    """
    TODO: Change num_topics to a parameter
    TODO: Make the stopwords consistent across all algorithms
    """
    # unpickle dataframe
    df = loadDataFrame(filename)

    # unpickle list of article Links
    keyword = filename[7:(len(filename)-5)]
    links = get_links(keyword)
    links.append('wikipedia')
    print("Length of links: ", len(links))

    # NMF is able to use tf-idf: a statistic intended to reflect how important a word is to a document in a collection or corpus
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=10000)
    tfidf = tfidf_vectorizer.fit_transform(df['data'])

    # Initialize data being saved in text file and adding the date
    to_save = []
    to_save.append(str(date.today()))

    # Run NMF Frobenius norm
    print("Running NMF with Frobenius norm...")
    to_save.append("\n\nRunning NMF with Frobenius norm...")
    nmf1 = NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

    # Print most relevant links per topic
    relevant_articles = get_doc_topics(nmf1, tfidf, links)
    relevant_links = get_relevant_links(relevant_articles, 2) # TODO: Change this to user input

    # Show word weights
    weights = get_weights(nmf1, tfidf_vectorizer)

    # Save results
    to_save.extend(print_topics(nmf1, tfidf_vectorizer, relevant_links))

    print("--------------------------------------------------------------------")

    # Run NMF with Kullback-Leibler
    print("Running NMF with Kullback-Leibler... ")
    to_save.append("\n\nRunning NMF with Kullback-Leibler... ")
    nmf2 = NMF(n_components=num_topics, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)

    # Print most relevant links per topic
    relevant_articles = get_doc_topics(nmf2, tfidf, links)
    relevant_links = get_relevant_links(relevant_articles, 2) # TODO: Change this to user input

    # Save results in a text file
    to_save.extend(print_topics(nmf2, tfidf_vectorizer, relevant_links))
    save_topics("./NMF-LDA_Results/"+keyword+".txt", to_save)

    return nmf1, tfidf_vectorizer;


def save_topics(filename, data):
    file = open(filename,"w")
    file.writelines(data)
    file.close()
