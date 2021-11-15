# encoding=utf-8
"""
gsGenSim runs a set of Word2Vec functions.
"""
import gensim.models
from gensim.models import Word2Vec, Doc2Vec
import gensim.downloader as api
import nltk
import gs_nlp_datadict as gsutil
import pandas as pd
from string import punctuation
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamulticore import LdaMulticore
from gensim.test.utils import datapath

# import pdb
# from sklearn.cluster import AffinityPropagation, SpectralClustering
# from sklearn.manifold import TSNE, SpectralEmbedding

import warnings
import parsing
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from gs_nlp_datadict import engstoplist

warnings.filterwarnings(action='ignore')
nltk.download('punkt', download_dir="/Users/bgh/dev/")
nltk.download('treebank', download_dir="/Users/bgh/dev/")
lyricfile = "/Users/bgh/dev/pydev/gsSentiment/rawdata/gs_corp.txt"
# gd_corpus = "/Users/bgh/dev/pydev/gsSentiment/gd_corpus.w2v"

def raw_preprocess(inp: str):
    """
    raw_preprocess contains ad-hoc methods to clean the raw data to be used as corpus
    each corpus will have generic plus special processing - the corp of Grateful Dead
    lyrics had verse and chorus formatting to clean:
            1. remove bracketed identifiers, ex. [Chorus]
            2. songs have their own vernacular - some of which I don't want to lose so
                I adapt grammar and spelling, such as allow the words 'aint' or 'hes'
            3. standardize musical words like ooh and la-la or 'truckin'.
            4. remove extra spaces, line breaks, quotes, apostrophe, parens
    Args:
        inp: str filename for raw text file used as corpus
    Returns: str of cleaned data
    """
    inp_post: str = ""
    bracks = re.compile('\[.+?\]')
    lyrics2 = bracks.sub()
    inp = lyrics2.replace("   ", "")
    for x in engstoplist:
        lyrics2 = lyrics2.replace("\n"," ")
    lyrics2 = lyrics2.lower()
    inp_post = parsing.remove_stopwords(lyrics2)
    return inp_post

def save_corpus(outf: str, out_str):
    """
    save_corpus simply writes a processed string back to file
    Args:
        txt_out: string which has been pre-processed, may extend this to save other nlp
        outputs to disk
    Returns: 0 if successful write to disk
    """
    with open(outf, mode='wt+', newline=None) as corpfile:
        corpfile.write(out_str)

    return corpfile.close()

def word_extraction(sentstr, stops):
    words = re.sub("[^\w]", " ", sentstr).split()
    cleaned_text = [w.lower() for w in words if w not in stops]
    new_s: str = ""
    for x in cleaned_text:
        new_s = new_s + " " + x
    return new_s

def do_cbow(word_tokens, min: int=1, siz: int=300, win: int=5):
    """
    submit a work tokenized corpus for vectorization with gensim c-bag of words
    :param word_tokens: list of list of str (words)
    :param min: minimum count of words during training
    :param siz: resulting vocab size
    :param win: size of 'neighborhood' of words
    :return:
    """
    cbow_model = Word2Vec(word_tokens, min_count=min, size=siz, window=win)

    return cbow_model

def do_skip_gram(word_tokens, min: int=1, siz: int=300, win: int=5, skipg: int=1):
    """
    create vectorized model based on skip gram
    :param word_tokens: submit corpus for vectorization with gensim skip-gram
    :param min:
    :param siz: vocab size
    :param win: size of 'neighborhood' of words
    :param skipg:
    :return:
    """
    skipg_model: gensim.models.KeyedVectors = Word2Vec(word_tokens, min_count=min, size=siz, window=win, sg=skipg)

    return skipg_model

def get_word_similarity(mdl, wrd1: str, wrd2: str):
    """
    given two two words and a model, display the vector similarity between them
    :param mdl: a gensim keyed vector model
    :param wrd1: str of word one
    :param wrd2: str of word two
    :return: similarity score, also prints result to console display
    """

    sim: float = mdl.similarity(wrd1, wrd2)
    print("Similarity between %s and %s: %.3f" % (wrd1, wrd2, sim))

    return sim
