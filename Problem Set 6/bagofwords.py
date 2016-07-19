import nltk
import re
import sys
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

def latdirall(content):
    lda = LatentDirichletAllocation(n_topics=10)
    tf_vectorizer = TfidfVectorizer(max_df=0.99, min_df=1,
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(content)
    lolz = lda.fit_transform(tf)
    tfidf_feature_names = tf_vectorizer.get_feature_names()
    return top_topics(lda, tfidf_feature_names, 10)

def top_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
    i = 1
    for topic in topics:
        print(i)
        print(topic)
        i += 1
    return topics

with open(sys.argv[1], 'r') as input_file:
    fil = input_file.read()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    topics = latdirall(fil.split('\n'))
