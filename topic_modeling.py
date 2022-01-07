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

import pandas as pd
from numpy import percentile, mean, median, std
from gs_datadict import OUTDIR, GS_CONTRACT, CORESTOPS
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.phrases import Phrases, bigram
from gensim.corpora import MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.test.utils import datapath
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# vsi = SentimentIntensityAnalyzer()
pd.options.plotting.backend = "plotly"
ldatopic_table = {"*": " * ", '"': ""}
topic_str_trans = str.maketrans(ldatopic_table)

def get_gdict_and_doc2bow(docs: list, at_least_docs: int=30):
    """
    prepare gensim terms dictionary and document-terms matrix as intermediate step in
    topic-modeling.
    my prep_lda was redundant with Fx's in gs_utils to word-tokenize a corpus for lda
    :param docs: should send list of list of words (word tokenized corpus)
    :return: list and gsm.Dictionary with term frequencies in Tweets and corpus
    """

    doc_dict: Dictionary = Dictionary(docs)

    # Filter out words that occur less than 2 documents, or more than 30% of the documents.
    doc_dict.filter_extremes(no_below=at_least_docs, no_above=0.5)

    doc_term_matrix: list = [doc_dict.doc2bow(doc) for doc in docs]

    return doc_term_matrix, doc_dict

def run_lda_model(doc_term, term_dict, topics: int = 30, chunk: int = 2000, eval: int=5,
                  train_iter: int = 100, word_topics=False):
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
    ldamodel: Lda = LdaModel(corpus=doc_term, num_topics=topics, id2word=term_dict,
                        chunksize=chunk, iterations=train_iter, passes=4, eval_every=eval,
                        per_word_topics=word_topics, alpha='auto')

    return ldamodel

def print_lda_topics(model: LdaModel, ntopics: int=25, nwords: int=6):
    """
    shows results of Latent Dirichlet Analysis topic modeling with gensim
    :param model: instance of gsm.models.ldamodel.LdaModel
    :param ntopics: number of topics to display
    :param nwords: number of words per topic to display
    :return:
    """
    ttopics: list = []
    for i, topic in model.show_topics(formatted=True, num_topics=ntopics, num_words=nwords):

        topic = str(topic).translate(topic_str_trans)
        ttopics.append(topic)
        print(str(i) + ": " + topic)
        print()

    return ttopics

def save_lda_model(ldam: LdaModel, mdl_f: str = "lda_model", ldadir: str=OUTDIR):
    """
    save a gensim.models.ldamodel.LdaModel instance to a file
    :param ldam: gensim lda model
    :param mdl_f: file name to save as
    :param ldadir: folder to save LDA models
    :return: None
    """
    temp_file = datapath(ldadir + mdl_f)
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

def get_lda_model_topictable(ldam: LdaModel, topics: int=20, wrds: int=8):
    """
    review lda models topics and terms, by passing them to pandas for ease of use
    :param ldam: trained LDA model
    :param topics: how many to display
    :param wrds: how many per topic
    :return: vectors for new content
    """
    pd.options.display.width = 110
    pd.options.display.max_columns = 13
    df = pd.DataFrame([[word for rank, (word, prob) in enumerate(words)]
                  for topic_id, words in ldam.show_topics(formatted=False, num_words=wrds, num_topics=topics)])

    print(df.head())

    return df

def update_model_new_txt(ldam: LdaModel, wrd_tokens: list, ldadict: Dictionary):
    """
    run new content through training with an existing model
    :param ldam: gensim.models.LdaModel
    :param wrd_tokens: list of word tokens for new tweets
    :param ldadict: Dictionary of type gensim.corpora.Dictionary
    :return:
    """
    if isinstance(wrd_tokens, list):
        if isinstance(wrd_tokens[0], list):
            # each tweet is a list of strings within the larger list
            if isinstance(wrd_tokens[0][0], str):
                for twt in wrd_tokens:
                    new_corpus = [ldadict.doc2bow(wrd) for wrd in twt]
        elif isinstance(wrd_tokens[0], str):
            # each list element is text of tweet, need to word tokenize...
            for twt in wrd_tokens:
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

def create_phrase_model(sents, min_occurs: int=10, delim: str="-"):
    """
    use gensim phrases to identify top bigrams in corpus
    :param sents: list of word-tokenized lyrics
    """
    phrase_mdl = Phrases(sents, min_count=min_occurs, delimiter=delim)

    return phrase_mdl

def test_phrase_model(test_lyrs, phrase_mdl: Phrases):
    """
    run a test set against an existing phrase model
    :param test_lyrs: word-tokenized lyrics
    :param phrase_mdl: a gensim Phrases model
    """
    found_phrases: dict = {}
    for phrase, score in phrase_mdl.find_phrases(test_lyrs).items():
        found_phrases[phrase] = score
        print(phrase, score)

    for sent in phrase_mdl[test_lyrs]:
        pass

    return found_phrases

def add_to_phrases(sents, p_mdl):
    """
    apply new unseen sentences to existing phrase model
    :param sents: lists of word-tokenized lists
    return: updated p_mdl
    """
    p_mdl.add_vocab(sents)

    return p_mdl

def get_phrase_stats(pmodel: Phrases):
    """
    export phrases and then run mean, median and std on phrase scores
    sort a dictionary of phrases and values by descending value
    :param pmodel: instance of a gensim Phrases model
    """
    phdict: dict = pmodel.export_phrases()
    statsdct: dict = {'mean': mean(list(pmodel.values()))}

    return