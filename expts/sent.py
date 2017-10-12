from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.path.append("..")

sys.dont_write_bytecode = True
import pickle
from utils.lib import O, Memoized
import numpy as np
from collections import OrderedDict
from network.mine import cite_graph, Miner
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from db import mysqldb
import pandas as pd
from sklearn.feature_extraction import text
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from expts.settings import dend as dend_settings
import cPickle as cPkl
from sklearn.externals import joblib
from utils.sk import bootstrap, qDemo

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

GRAPH_CSV = "../data/citemap_v10.csv"

# For 11 TOPICS
ALPHA = 0.22359
BETA = 0.53915
ITERATIONS = 100

TOPICS_ALL = ["Program Analysis", "Requirements", "Metrics", "Applications",
                            "Performance", "Miscellaneous", "Testing", "Source Code",
                            "Architecture", "Modeling", "Developer"]

TOPIC_THRESHOLD = 3

# Global Settings
THE = O()
THE.permitted = "all"    # conference/journal/all
THE.version = "v4"
THE.use_numeric = False

STOP_WORDS = text.ENGLISH_STOP_WORDS.union(['software', 'engineering', 'paper', 'study', 'based',
                                                                                        'results', 'approach', 'case', 'workshop', 'international', 'research',
                                                                                        'conference', 'introduction', 'editors', 'article', 'issue', 'month',
                                                                                        'copyright', 'special', 'used', 'using', 'use', 'studies', 'review',
                                                                                        'editorial', 'report', 'book', 'ieee', 'published', 'science', 'column',
                                                                                        'author', 'proposed', 'icse', 'article', 'year', 'articles', 'page', '2000',
                                                                                        '2004', 'papers', 'computer', 'held', 'editor'])


COLORS_ALL = ["lightgray", "red", "blue", "darkslategray",
                            "yellow", "darkmagenta", "cyan", "saddlebrown",
                            "orange", "lime", "hotpink"]


MIN_DIVERSITY_SCORE = 0.075

def mkdir(directory):
    """
    Implements the "mkdir" linux function
    :param directory:
    :return:
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def get_n_topics():
    if THE.permitted == "journals":
        return 7
    if THE.permitted == "all":
        return 11

@Memoized
def retrieve_graph_lda_data():
    """
    Fetch stored metadata
    :return:
    """
    graph_file = 'cache/%s/%s/graph.pkl' % (THE.version, THE.permitted)
    vectorizer_file = 'cache/%s/%s/vectorizer.pkl' % (THE.version, THE.permitted)
    doc_2_vec_file = 'cache/%s/%s/doc_2_vec.pkl' % (THE.version, THE.permitted)
    documents_file = 'cache/%s/%s/documents.pkl' % (THE.version, THE.permitted)
    lda_model_file = 'cache/%s/%s/lda_model.pkl' % (THE.version, THE.permitted)
    vocabulary_file = 'cache/%s/%s/vocabulary.pkl' % (THE.version, THE.permitted)
    if os.path.isfile(graph_file) and os.path.isfile(vectorizer_file) \
                    and os.path.isfile(doc_2_vec_file) and os.path.isfile(documents_file) \
                    and os.path.isfile(lda_model_file) and os.path.isfile(vocabulary_file):
        with open(graph_file) as f:
            graph = cPkl.load(f)
        miner = Miner(graph)
        with open(vectorizer_file) as f:
            miner.vectorizer = cPkl.load(f)
        with open(doc_2_vec_file) as f:
            miner.doc_2_vec = joblib.load(f)
        with open(documents_file) as f:
            miner.documents = cPkl.load(f)
        with open(lda_model_file) as f:
            lda_model = cPkl.load(f)
        with open(vocabulary_file) as f:
            vocab = cPkl.load(f)
    else:
        miner, graph, lda_model, vocab = store_graph_lda_data()

    return miner, graph, lda_model, vocab

def store_graph_lda_data():
    miner, graph, lda_model, vocab = get_graph_lda_data()
    folder_name = 'cache/%s/%s' % (THE.version, THE.permitted)
    mkdir(folder_name)
    with open('cache/%s/%s/graph.pkl' % (THE.version, THE.permitted), 'wb') as f:
        cPkl.dump(graph, f, cPkl.HIGHEST_PROTOCOL)
    with open('cache/%s/%s/vectorizer.pkl' % (THE.version, THE.permitted), 'wb') as f:
        cPkl.dump(miner.vectorizer, f, cPkl.HIGHEST_PROTOCOL)
    with open('cache/%s/%s/doc_2_vec.pkl' % (THE.version, THE.permitted), 'wb') as f:
        joblib.dump(miner.doc_2_vec, f)
    with open('cache/%s/%s/documents.pkl' % (THE.version, THE.permitted), 'wb') as f:
        cPkl.dump(miner.documents, f, cPkl.HIGHEST_PROTOCOL)
    with open('cache/%s/%s/lda_model.pkl' % (THE.version, THE.permitted), 'wb') as f:
        cPkl.dump(lda_model, f, cPkl.HIGHEST_PROTOCOL)
    with open('cache/%s/%s/vocabulary.pkl' % (THE.version, THE.permitted), 'wb') as f:
        cPkl.dump(vocab, f, cPkl.HIGHEST_PROTOCOL)
    return miner, graph, lda_model, vocab

def get_graph_lda_data():
    graph = cite_graph(GRAPH_CSV)
    miner = Miner(graph, THE.permitted)
    lda_model, vocab = miner.lda(get_n_topics(), n_iter=ITERATIONS, alpha=ALPHA, beta=BETA, stop_words=STOP_WORDS)
    return miner, graph, lda_model, vocab

def format(val):
    try:
        fval = float(val)
        if fval.is_integer():
            return int(fval)
        else:
            return fval
    except ValueError:
        return val


def paper_text(graph):
    # getting title and abstract together per paper
    p_text = {}
    p_title_only = []
    for paper_id, paper in graph.get_paper_nodes(THE.permitted).items():
        if paper.abstract != "None":
            p_text[paper_id] = paper.title
            p_title_only.append(paper_id)
        else:
            p_text[paper_id] = paper.title + ". " + paper.abstract
    return p_text, p_title_only


def sentiment_count(res):
    #sentiment count per paper
    text_sentiment = []
    sc = [0, 0, 0]
    for s in res["sentences"]:
        s_val = format(s["sentimentValue"])
        entry = (" ".join([t["word"] for t in s["tokens"]]), s_val)
        if entry[0] != u'None':
            text_sentiment.append(entry)
        if s_val == 1 or s_val == 0:
            sc[0] += 1
        elif s_val == 3 or s_val == 4:
            sc[2] += 1
        else:
            sc[1] += 1
    s_label = sc.index(max(sc))
    return s_label, text_sentiment


def sentiment_papers(p_text):
    sent_papers = {}
    p_t_sentiment = {}
    count = [0, 0, 0]
    for paper_id, paper_text in p_text.items():
        res = nlp.annotate(paper_text,
                           properties={
                               'annotators': 'sentiment',
                               'outputFormat': 'json',
                           })

        s_label, t_sentiment = sentiment_count(res)
        p_t_sentiment[paper_id] = t_sentiment
        sent_papers[paper_id] = s_label
        if s_label == 0:
            count[0] += 1
        elif s_label == 1:
            count[1] += 1
        else:
            count[2] += 1
    print(count)
    return p_t_sentiment, sent_papers


def pickle_operating(fname, item):
    # save or load the pickle file.
    file_name = '%s.pickle' % fname
    print(file_name)
    if not item:
        with open(file_name, 'rb') as fs:
            item = pickle.load(fs)
            return item
    else:
        with open(file_name, 'wb') as fs:
            pickle.dump(item, fs, protocol=pickle.HIGHEST_PROTOCOL)


def topic_paper(graph, miner):
    # getting the topic per paper.
    topic_papers = {}
    for paper_id, paper in graph.get_paper_nodes(THE.permitted).items():
        topics = miner.documents[paper_id].topics_count
        topic = topics.argmax()
        if topic not in topic_papers.keys():
            topic_papers[topic] = []
        topic_papers[topic].append(paper_id)
    return topic_papers


def topics_sentiment(topic_papers, sent_papers):
    # getting the sentiment count per topic
    topics_paper_sent = {}
    topics_sent = {}
    for topic, papers_id in topic_papers.items():
        topics_paper_sent[topic] = []
        topics_sent[topic] = [0, 0, 0]
        for p_id in papers_id:
            topics_paper_sent[topic].append((p_id, sent_papers[p_id]))
            topics_sent[topic][sent_papers[p_id]] += 1
    return topics_paper_sent, topics_sent


def tprint(topic, topics_id_sent):
    for topic_id, topic_sent in topics_id_sent.items():
        print("%-20s\t: (Pos: %s, Neutral: %s, Neg: %s)" % (topic[int(topic_id)], topic_sent[0], topic_sent[1], topic_sent[2]))


if __name__ == "__main__":
    miner, graph, lda_model, vocab = retrieve_graph_lda_data()
    p_text, p_title_only = paper_text(graph)
    p_t_sentiment, sent_papers = sentiment_papers(p_text)
    pickle_operating("full_text_sent", p_t_sentiment)
    pickle_operating("paper_sent", sent_papers)
    topic_papers = topic_paper(graph, miner)
    topics_paper_sent, topics_sent = topics_sentiment(topic_papers, sent_papers)
    pickle_operating("topics_paper_sent", topics_paper_sent)
    pickle_operating("topics_sent", topics_sent)
    tprint(TOPICS_ALL, topics_sent)