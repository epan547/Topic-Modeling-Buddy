import pickle
import nltk
from os import path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

def loadDataFrame():
    userin = ((input("enter a topic: ")).lower()).replace(" ", "-")
    filename = "./topics/" + userin + "-content.pckl"

    if path.exists(filename):
        infile = open(filename, "rb")
        df = pickle.load(infile)
        print(type(df))
        print(df)
        infile.close()
        return df
    else:
        return ("this topic has not yet been pickled. run wiki.py first.")

def print_topics(model, count_vectorizer, n_top_words=5):
    words = count_vectorizer.get_feature_names()
    for index, topic in enumerate(model.components_):
        print("\nTopic #%d: ", index)
        print(" ".join([words[i]
            for i in topic.argsort()[:-n_top_words -1:-1]]))

def runLDA(df, number_topics=10):
    # Check .components_ for lda, to get
    count_vectorizer = CountVectorizer(stop_words='english')

    count_data = count_vectorizer.fit_transform(df['Content'])

    lda = LDA(n_components=number_topics, n_jobs=-1)

    lda.fit(count_data)
    normalized_by_column = lda.components_ / lda.components_.sum(axis=0)
    print(count_vectorizer.get_feature_names()[normalized_by_column[0,:].argmax())])
    print_topics(lda, count_vectorizer)

if __name__ == "__main__":
    df = loadDataFrame()
    runLDA(df)
