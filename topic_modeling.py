# coding=utf-8
"""
gs_topic_models uses Latent Dirichlet Allocation (LDA) to identify likely topics in the
text of tweets.  each Tweet corresponds to a 'Document'.

uses flow established in gs_Tweet_main, the controlling script for the Twitter analytics app
- we want to capture the text of tweets across the current 3 topic datasets:
    Gamestop, Superleague, and FacebookMeta
- I can place a save point after running the duplicate/retweet filter with each of the
three topic datasets.
- In the control script for each topic dataset, I will ave each list of dict as a
JSON file.  Then, in this script, I read one or more of the archive files into a corpus
to use for topic modeling
- we want a large, heterogenous Tweet 'corpus' for topic modeling, knowing which topic dataset
each tweet came from can be helpful.  LDA topic modeling is an unsupervised approach, but by
knowing the origin of each tweet, we can look at how closely LDA correlates with the topic
datasets (lack of correlation could point to problems with my topic dataset building)

"""

import os
import pandas as pd
import gs_utils as gsutil
from gs_datadict import OUTDIR, MODELDIR, GS_STOP, STOP_LYRICS
from string import punctuation
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

pd.options.plotting.backend = "plotly"

def prep_for_lda(twl: list):
    """
    preprocess cleaning for LDA topic modeling
    string package punctuation includes !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    :param twl: list of dict of tweets
    :return: list of str with scrubbed tweet text
    """
    stop = set(stopwords.words('english'))
    exclude = set(punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        """
        inner Fx to lowercase tweet, strip punctuation and remove stops
        :param doc: list of tweet text strings
        :return:
        """
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = "".join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean: list = []
    for tw in twl:
        if isinstance(tw, dict):
            doc_clean.append(clean(tw['text']).split())
        elif isinstance(tw, str):
            doc_clean.append(clean(tw).split())
        else:
            print("ERROR- prep_for_lda expects list of dict or list of str for Tweets")
            return None

    return doc_clean

def gensim_doc_terms(docs: list):
    """
    prepare gensim terms dictionary and document-terms matrix as intermediate step in
    topic-modeling of Tweet dataset.
    :param docs: list of tweets
    :return: list and gsm.Dictionary with term frequencies in Tweets and corpus
    """
    doc_dict: Dictionary = Dictionary(docs)
    doc_term_matrix: list = [doc_dict.doc2bow(doc) for doc in docs]

    return doc_term_matrix, doc_dict

def run_lda_model(doc_term, term_dict, topics: int = 5, chunk: int = 3000,
                  train_iter: int = 50, word_topics=False):
    """
    run training cycles with doc-term matrix and docterms dictionary to create a
    gensim LdaModel for topic modeling
    parameters info and other details can be found at
        https://radimrehurek.com/gensim/models/ldamodel.html

    :param doc_term: doc-term matrix created in gensim_doc_terms
    :param term_dict: document term dictionary created in gensim_doc_terms
    :param topics: number of topics to compute and display
    :param chunk: number of documents used in each training 'chunk' def=2000
    :param train_iter: cycles of training to run in gensim
    :param word_topics: value for per_word_topics parm, True computes list of topics in descending order of prob
    :return:
    """
    Lda = LdaModel
    ldamodel: Lda = Lda(corpus=doc_term, num_topics=topics, id2word=term_dict,
                        chunksize=chunk, iterations=train_iter, passes=3,
                        update_every=0, per_word_topics=word_topics, alpha='auto',
                        eta=None, minimum_probability=0.01)

    return ldamodel

def display_lda(model: LdaModel, ntopic: int = 5):
    """
    shows results of topic modeling analysis with gensim
    :param model: instance of gsm.models.ldamodel.LdaModel
    :param ntopic: number of topics to display
    :return:
    """
    print(model.print_topics(num_topics=ntopic, num_words=3))

    return None

def save_lda_model(ldam: LdaModel, mdl_f: str = "lda_model"):
    """
    save a gensim.models.ldamodel.LdaModel instance to a file
    :param ldam: gensim lda model
    :param mdl_f: file name to save as
    :return: None
    """
    temp_file = datapath(OUTDIR + mdl_f)
    ldam.save(temp_file)

    return None

def load_lda(fq_mdl: str = "lda_model"):
    """
    load a pretrained gensim lda model from file
    :param fq_mdl: str filename of saved LDA model, default path is ./OUTDIR
    :return: lda model
    """
    Lda = LdaModel
    ldam = Lda.load(OUTDIR + fq_mdl)

    return ldam

def test_text_with_model(new_txt: str, ldam: LdaModel, docterm):
    """
    get vectors for new content using our pretrained model
    :param new_txt: new tweet not part of training set
    :param ldam: trained LDA model
    :param docterm: gensim doc-term matrix
    :return: vectors for new content
    """

    vecs = ldam.update(new_txt, update_every=0)

    return vecs

def update_model_new_txt(ldam: LdaModel, wrd_tokns: list, ldadict: Dictionary):
    """
    run new content through LDA modeling
    :param ldam: gensim.models.LdaModel
    :param wrd_tokns: list of word tokens for new tweets
    :param ldadict: Dictionary of type gensim.corpora.Dictionary
    :return:
    """
    if isinstance(wrd_tokns, list):
        if isinstance(wrd_tokns[0], list):
            # each tweet is a list of strings within the larger list
            if isinstance(wrd_tokns[0][0], str):
                for twt in wrd_tokns:
                    new_corpus = [ldadict.doc2bow(wrd) for wrd in twt]
        elif isinstance(wrd_tokns[0], str):
            # each list element is text of tweet, need to word tokenize...
            for twt in wrd_tokns:
                tmp_tok: list = str(twt).split()
                new_corpus = [ldadict.doc2bow(wrd) for wrd in tmp_tok]

        new_content = new_corpus[0]
        vector = ldam[new_content]

    return vector

def get_model_diff(lda1: LdaModel, lda2: LdaModel):
    """
    get differences between pairs of topics from two models
    :param lda1:
    :param lda2:
    :return:
    """
    m1 = LdaMulticore.load(datapath(lda1))
    m2 = LdaMulticore.load(datapath(lda2))

    mdiff, annotation = m1.diff(m2)
    topic_diff = mdiff

    return topic_diff

def get_top_terms(lda: LdaModel, dDict: Dictionary, tpcs: int = 5, tpc_trms: int = 8):
    """
    map word index to actual word when printing top terms for topics
    :param lda: gensim.models.LdaModel instance
    :param dDict: gensim.corpora.Dictionary instance
    :param tpcs: number of topics for which to show terms
    :param tpc_trms: number of top terms to show for each topic
    :return:
    """
    tmpDict = dDict.id2token.copy()

    for x in range(tpcs):
        ttrm: list = lda.get_topic_terms(topicid=x, topn=tpc_trms)
        print("\ntop terms for topic %d:" % x)
        for twrd, val in ttrm:
            print("%s has probability %.3f" % (tmpDict[twrd], val))

    return None
