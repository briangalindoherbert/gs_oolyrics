# encoding=utf-8
"""
gsGenSim runs a set of Word2Vec functions.
"""
from __future__ import generator_stop

import re
from os import listdir
from os.path import join, isfile, isdir
from math import log
from numpy import percentile, mean, median, std
import gensim.corpora
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.stem.wordnet import WordNetLemmatizer
from gs_datadict import LYRICDIR, CORESTOPS, GS_CONTRACT, CHAR_CONVERSION, QUOTEDASH_TABLE
from gs_utils import retrieve_genre_registry, get_artists_in_genre, clean_name
from gs_genre_class import Genre_Aggregator
# import nltk
# from sklearn.model_selection import train_test_split

unic_conv_table = str.maketrans(CHAR_CONVERSION)
punc_tbl = str.maketrans({key: None for key in "[];:,.?!*$&@%<>'(){}"})
re_brkttxt = re.compile(r"\[.+\]")
lemma = WordNetLemmatizer()

def get_tag_source(dtag: str, genre: str='rock', registry: list=None):
    """
    given a TaggedDocument tag, get the name of the song and its lyrics, returned as a
    dict.  Important for comparing test results to source lyrics.
    tags = artist name (no spaces) plus 3-digit zero-filled counter (ex. '005')
    lyrics are found by counting end-of-track markers in source file- first song is '000'
    :param dtag: TaggedDocument tag, consisting of artist name + 3 digit track counter
    :param genre: need this to construct filename for artist's lyrics file
    :param registry: optional list of artist tracks dictionaries- can append song title
    :return: dict with keys: 'artid', 'artname', 'tracknum', 'lyrics'
    """
    art = dtag[:-3]
    traknum = dtag[-3:]
    trakint = int(traknum)

    art_trakdct: dict = {'art_name': art, 'genre': genre, 'tracknum': traknum}
    if registry:
        for artreg in registry:
            if dtag in artreg.keys():
                art_trakdct['track'] = artreg[dtag]

    lyrfile = genre + "_" + art + ".lyr"
    fqf = join(LYRICDIR, lyrfile)
    if isfile(fqf):
        eot_counter: int = 0
        trak_lines: list = []
        word_ct: int = 0
        with open(fqf, mode="r") as f_h:
            for line in f_h:
                if eot_counter == trakint:
                    if len(line) > 5:
                        linetok, word_ct = line_cleanup(line, word_ct)
                        trak_lines.append(linetok)
                if line.startswith("SONGBREAK"):
                    eot_counter += 1
                    if eot_counter > trakint:
                        break
        if word_ct:
            print("pulled %d words of lyrics for %s" % (word_ct, art))
            art_trakdct['lyrics'] = trak_lines

    return art_trakdct

def create_load_tagged_docs(artist_file, folder: str = LYRICDIR):
    """
    loads lyrics into memory, each song as a list of word tokens and passed to TaggedDocument
    along with a unique integer tag.
    :param artist_file: str or list for single artist file to select
    :param folder: str folder name
    :return list of str corpus
    """
    agg_wrds: int = 0
    doc_wrds: int = 0
    trak: list = []
    doclst: list = []
    tagint: int = 0
    for fname in listdir(folder):
        if fname in artist_file:
            fqfil = join(folder, fname)
            f_h = open(fqfil, mode="r")
            for line in f_h:
                if len(line) > 5:
                    if line.startswith("SONGBREAK"):
                        doclst.append(TaggedDocument(trak, [tagint]))
                        print("added tagged doc id %d with %d words" % (tagint, agg_wrds))
                        trak = []
                        tagint += 1
                        doc_wrds = 0
                        continue
                    else:
                        doc_wrds, splts = line_cleanup(txt=line, word_ct=doc_wrds)
                        agg_wrds += doc_wrds
                        trak.extend(splts)
            print("processed %d words from %s" % (agg_wrds, fname))
            f_h.close()

    return doclst

def do_skip_gram(iterlyr, batchsiz: int = 5000, passes: int = 5, thrds: int = 4, maxv: int = 600,
                 dim: int = 64, grpsz: int = 5, skipg: int = 1):
    """
    create vectorized model based on skip gram, some parms that are settable with word2vec
    but are not settable from this Fx's parms: hs- if 1 use hierarchical softmax,
    default is negative sampling- see w2v 'ns' parm,
    max_vocab_size - while training- as opposed to max_final_vocab for output model
    :param iterlyr: batches of lyrics, provided by an iterator tuned to batchsiz
    :param batchsiz: words provided per cycle to worker instances, typically 5-10k
    :param passes: number of passes (epochs) through the corpus, default=3
    :param thrds: number of concurrent processes to run (multi-processing threads)
    :param maxv: maximum final vocabulary for model
    :param dim: dimensionality of resulting word vectors (vector_size)
    :param grpsz: max distance from current to predicted word, typically 1-7
    :param skipg: 1 selects skipgram, 0 or other selects cbow
    :return:
    """

    skipg_mdl = Word2Vec(sentences=iterlyr, batch_words=batchsiz, epochs=passes,
                         vector_size=dim, window=grpsz, workers=thrds, sg=skipg, hs=1,
                         max_final_vocab=maxv, sample=1e-3)

    return skipg_mdl

def document_skipgram(itertrax, passes: int=8, grpsize: int=8, dim: int=80, dm: int=1,
                      thrds: int=3, alph=0.025):
    """
    wrapper Fx to call Doc2Vec training (pass a MusicalMeg generator-iterator to this).
    parameters similar to word2vec.
        NOTE: as in w2v, either specify min_count OR max_final_vocab
      PV-DM is analogous to Word2Vec CBOW - predict center word by averaging doc vec and
        word vectors.
      PV-DBOW is analogous to Word2Vec SG - doc vectors trained on predicting a task of
        predicting a target word. often combines doc vec with word vecs to predict a
        neighboring word.
    :param itertrax: iterator to pass music tracks to doc2vec
    :param passes: epochs or 'passes' through the documents to make during training
    :param grpsize: int size of cbow window or skipgram distance in training evaluation
    :param dim: number of dimensions for resulting doc vectors
    :param dm: int 1 for dm (equiv to w2v cbow) or 0 for dcbow (equiv to w2v skipgram)
    :param thrds: number of concurrent workers, or threads, during training
    :param alph: learning rate, floating point num
    :return: Doc2Vec model - a container with document vectors (dv).
    """

    model = Doc2Vec(itertrax, vector_size=dim, window=grpsize, epochs=passes, negative=5,
                    dm=dm, min_count=2, workers=thrds, dbow_words=1, alpha=alph)
    # use dm=1 for cbow-like method, or 0 for skipgram-like method

    return model

def test_sentence_doc2vec(sentence: list, d2v: Doc2Vec, topsiz: int=10):
    """
    test word-tokenized sentences against a doc2vec model to get 10 most similar docs

    :param sentence: list with each element a cleaned, word-tokenized list
    :param d2v: a trained doc2vec model
    :param topsiz: number of most similar songs to show
    :return:
    """
    test_results: list = []
    infer2 = d2v.infer_vector(sentence)
    mostsim = d2v.dv.most_similar([infer2], topn=topsiz)
    print("  ---- test for %d most similar docs ----" % topsiz)
    for i in range(topsiz):
        # test_results.append(get_tag_source(dtag=sim1[i][0], genre=test_genre))
        test_results.append(mostsim[i][0])
        print(mostsim[i])

    return test_results

def get_doc2vec_features(d2v: Doc2Vec, wordlistA, wordlistB, questions):
    """
    show off features and methods in gensim 4.0 doc2vec models
    :param d2v: the unsupervised trained model from tracks of lyrics
    :param wordlistA: list of sentences
    :param wordlistB: list of sentences like a above
    :param questions: questions for logic answers from model
    :return:
    """
    wordA = wordlistA[0]
    wordB = wordlistA[1]
    pairs: list = []
    for w1, w2 in zip(wordlistA, wordlistB):
        pairs.append((w1, w2))

    d2v.dv.most_similar(wordA)
    d2v.dv.most_similar_cosmul(wordB)
    d2v.dv.wmdistance(wordlistA, wordlistB)
    d2v.dv.similar_by_word(wordA)
    d2v.dv.similar_by_vector(wordA)
    d2v.dv.doesnt_match(wordlistA)
    d2v.dv.similarity(wordA, wordB)
    d2v.dv.n_similarity(wordlistA, wordlistB)
    d2v.dv.evaluate_word_pairs(pairs)
    d2v.dv.evaluate_word_analogies(questions)
    # d2v.dv.log_accuracy(section)

    return

def doc_model_phrases(lyrobj, testsents):
    """
    uses the phrases feature of the gensim doc2vec model
    :param lyrobj: object instance of TaggerTed, SinginSue, or MusicalMeg Class
    :param testsents: list of tokenized test sentences to use on trained phrase model
    :return:
    """
    from gensim.models.phrases import Phrases
    phrase_model = Phrases(sentences=lyrobj, min_count=4, delimiter="-", progress_per=4000)
    # Apply the trained model to each sentence of a corpus, using the same [] syntax:

    for sent in phrase_model[testsents]:
        pass
    # Update the model with two new sentences on the fly.
    phrase_model.add_vocab([["hello", "world"], ["meow"]])

    for phrase, score in phrase_model.export_phrases(lyrobj, out_delimiter="-"):
        print(" phrase: %s  score: %.3f" % phrase, score)

    return

def get_lyrs_extra_training(ttobj, artst, prefx):
    """
    pull lyrics and tags for an artist, for additional doc2vec training
    :param ttobj:
    :param artst:
    :return:
    """
    from gs_genius_wrappers import load_art_lyrics

    tfidfdct: dict = ttobj.tfidf_artst[artst]
    thisreg: dict = ttobj.trax[artst]
    td_lyrs: list = []

    for wrds, tags in load_art_lyrics(prefix=prefx, artst=artst, artreg=thisreg):
        tmp: list = []
        for wrd in wrds:
            if tfidfdct.get(wrd):
                if tfidfdct[wrd] > 0.0299:
                    tmp.append(wrd)
                else:
                    continue
            else:
                continue
        td_lyrs.append(TaggedDocument(tmp, [tags]))

    return td_lyrs

def line_cleanup(txt: str, word_ct: int, wrd_tok: bool=True):
    """
    performs common formatting for lines of text including the following:
    1. convert to lower case
    2. expand contractions as per GS_CONTRACT from data_dict file
    3. convert common extended ascii/unicode chars like accented quote marks or ellipsis
    4. remove bracketed text which is common for annotations within lyrics
    5. remove punctuation
    6. remove stopwords from list CORESTOPS also from data_dict file
    :param txt: string to be wrangled
    :param word_ct: int running total of words, will be incremented and returned by Fx
    :param wrd_tok: bool default to return list of word tokens, else return a string
    :return: tokenized list of words
    """

    if isinstance(txt, str):
        # encd = txt.encode(encoding='ascii', errors='ignore')
        # txt = encd.decode(encoding='utf8', errors='ignore')
        # tmp = txt.lower()
        tmp = txt.lower().translate(unic_conv_table)
        tmp = tmp.translate(QUOTEDASH_TABLE)
        # nuclear option for trimming extra unicode chars:
        # binstr = tmp.encode("ascii", "ignore")
        # tmp = binstr.decode()
        # remove number at end of song inserted by genius
        tmp = re.sub(r"([a-z])\d+\n", repl=r"\1\n", string=tmp)
        for wrd, expand in GS_CONTRACT.items():
            tmp = re.sub(wrd, repl=expand, string=tmp)
        tmp = re_brkttxt.sub(repl="", string=tmp)
        tmptrans = tmp.translate(punc_tbl)
        splts: list = tmptrans.split()
        linetok: list = [w for w in splts if w not in CORESTOPS]
        word_ct += len(linetok)
    else:
        print("line cleanup expects an input string")
        linetok = []
        word_ct = 0

    if wrd_tok:
        return linetok, word_ct
    else:
        tmp = " ".join([w for w in linetok])
        return tmp, word_ct

def feed_specific_artists(artlst=None, srcdir: str = LYRICDIR):
    """
    adapted from feed_sent, this generator allows a slimmed-down corpus from a list of
    artists- this is for my doc2vec modeling needs and is used by TaggerTed class __iter__
    currently, artlst must be name of valid artist .lyr file or files, need to enhance
    this to accept list of artists too
    :param artlst: list of artists
    :param srcdir: usually the LYRICDIR
    :return:
    """
    agg_wrds: int = 0
    eottag: bool = False
    trak_wrds: list = []
    if isinstance(artlst, str):
        artlst = [artlst]

    for fname in listdir(srcdir):
        if fname in artlst:
            fqf = join(srcdir, fname)
            if isfile(fqf):
                sep_loc = fname.find("_")
                dot_loc: int = fname.find(".")
                art_nam: str = fname[sep_loc + 1: dot_loc]
                f_h = open(fqf, mode="r", encoding="utf-8", newline="")
                for line in f_h:
                    if len(line) > 5:
                        if line.startswith("SONGBREAK"):
                            eottag = True
                        else:
                            linetok, agg_wrds = line_cleanup(txt=line, word_ct=agg_wrds)
                            trak_wrds.extend(linetok)

                        if eottag:
                            yield trak_wrds, art_nam
                            trak_wrds = []
                            eottag = False

                print("    ---- finished streaming %s ----" % art_nam)
                f_h.close()

    return

def feed_sent(folder, prefix, eot: bool = False):
    """
    this generator returns each non-blank line in lyrics file as stream to the object
    it also extracts a list of artists, and calls line_cleanup for text wrangling
    if eot is True, it sends a flag at the end of each track so that the calling method
    can process tracks of lyrics (this is left to the object __iter__ method to aggregate,
    as individual lines are needed for word2vec but tracks for doc2vec.
    :param folder: folder where lyric files live
    :param prefix: word before '_' in files denotes genre, can use 'all', genre str,
    or request multiple genres with a list
    :param eot: if true send an eotflag when end-of-track marker is read
    :return NO return value for generator Fx's
    """
    agg_wrds: int = 0
    eottag: bool = False

    for fname in listdir(folder):
        sep_loc = fname.find("_")
        pre_name = fname[:sep_loc]
        fqfil = join(folder, fname)
        if isfile(fqfil):
            # select genre as str for one or as list for multiple
            do_this: bool = True
            if isinstance(prefix, str):
                if pre_name != prefix:
                    do_this: bool = False
            elif isinstance(prefix, list):
                if pre_name not in prefix:
                    do_this: bool = False
            if do_this:
                # if it is a file with the correct prefix, stream it!
                dot_loc: int = fname.find(".")
                art_nam: str = fname[sep_loc + 1: dot_loc]
                f_h = open(fqfil, mode="r", encoding="utf-8", newline="")
                for line in f_h:
                    if len(line) > 5:
                        if line.startswith("SONGBREAK"):
                            if eot:
                                # end-of-track lets us mark songs for a TaggedDocument
                                eottag = True
                                wtok: list = []
                            else:
                                # do not process this line
                                continue
                        else:
                            wtok, agg_wrds = line_cleanup(line, agg_wrds)

                        yield wtok, art_nam, eottag
                        if eot and eottag:
                            eottag = False

                print("    ---- finished streaming %s ----" % art_nam)
                f_h.close()
    print("  -- Completed Epoch of Lyrics, %d words" % agg_wrds)
    return

class LyricalLou:
    """
    LyricLou is a class that generates sequences of words for a word2vec training algorithm
    """
    genres: list = ['rock', 'rap', 'firstwave', 'alternative', 'metal', 'punk', 'country',
                    'folkrock', 'pop']

    def __init__(self, gen: str = 'rock', lydir: str = LYRICDIR):
        if gen:
            if gen == "all":
                self.genre: list = LyricalLou.genres
            elif isinstance(gen, str):
                if gen in LyricalLou.genres:
                    self.genre: str = gen
                else:
                    print("Lyrical Lou says... %s is not a valid genre" % gen)
                    raise Exception
            elif isinstance(gen, list):
                self.genre: list = []
                for gr in gen:
                    if gr in LyricalLou.genres:
                        self.genre.append(gr)
                    else:
                        print("Lyrical Lou says... %s is not a valid genre" % gr)
        if lydir:
            if lydir == LYRICDIR:
                self.folder = lydir
            else:
                self.folder = lydir
        else:
            lydir = LYRICDIR
        self.corpbow = []
        self.corpdic: Dictionary = {}
        self.word_count = 0
        self.word_freq: dict = {}
        self.artists: list = []
        self.stream_by_trak: bool = False
        self.calc_word_freq()

    def __iter__(self):
        artist_cur: str = ""
        docsents: list = []
        artist_words: int = 0
        trak_ct: int = 0
        for lyrs, artst, tag in feed_sent(self.folder, self.genre, eot=self.stream_by_trak):
            if not artist_cur == artst:
                if not artist_cur == "":
                    print("    streamed %d words from %d tracks for %s" % (artist_words, trak_ct, artist_cur))
                artist_cur = artst
                trak_ct = 0
            if lyrs:
                docsents.extend(lyrs)
            if tag:
                # send list of list of word tokens plus 1-element doc_tag list for each song
                yield docsents
                artist_words += len(docsents)
                docsents = []
                trak_ct += 1

        return

    def __len__(self):
        return self.word_count

    def create_bow(self):
        """
        create a gensim corpora.Dictionary from the corpus for this object,
        then create Bag of Words corpus from the Dictionary
        persist the Dictionary and BoW as instance variables
        :return:
        """
        if self.corpdic:
            print("Dictionary and BoW already calculated for this object")
            print("    Dictionary Bag of Words is %d words long" % len(self.corpbow))
        else:
            if self.word_count != 0:
                # need word frequencies if we ain't got 'em
                self.calc_word_freq()
        lyric_corpus: list = []
        artist_lines: list = []
        total_wrds: int = 0
        cur_artist: str = ""
        for lyrs, artst, traktag in feed_sent(self.folder, self.genre):
            if not cur_artist.startswith(artst):
                if not cur_artist == "":
                    lyric_corpus.append(artist_lines)
                artist_lines = []
                cur_artist = artst
                self.artists.append(cur_artist)

            # aggregate lines for each artest, then append to lyric_corpus
            artist_lines.extend(lyrs)
            total_wrds += len(lyrs)

        # existence of word in word_freq ensures freq of at least 2
        lyriclines = [[wrd for wrd in line if self.word_freq.get(wrd)] for line in lyric_corpus]

        self.corpdic = gensim.corpora.Dictionary(lyriclines)
        self.corpbow = [self.corpdic.doc2bow(line) for line in lyric_corpus]
        print("create_bow complete: %d sentences" % len(self.corpbow))

        return

    def calc_word_freq(self, min_count: int = 2):
        """
        create a dictionary of words and their frequency, configured to be called once
        also parses any words with a count of less than 2
        :param min_count: only include words that occur at least this many times
        :return:
        """
        if self.word_count != 0:
            print("calc_word_freq: already has %d words" % self.word_count)
        else:

            for lyrs, wrdsinc, artstinc in feed_sent(self.folder, self.genre):
                # tok_line: list = [self.lemma.lemmatize(w) for w in lyrs]
                for wrd in lyrs:
                    if wrd not in self.word_freq:
                        # unique (new) words...
                        self.word_count += 1
                        self.word_freq[wrd] = 1
                    else:
                        # repeat words...
                        self.word_freq[wrd] += 1

            self.word_freq = {k: v for k, v in self.word_freq.items() if not v < min_count}
            self.word_freq = {k: v for k, v in sorted(self.word_freq.items(),
                                                      key=lambda item: item[1], reverse=True)}
            print("calc_word_freq: %d words with min_count %d" % (len(self.word_freq), min_count))

        return

class TaggerTed(LyricalLou):
    """
    TaggerTed is a Class that inherits from LyricalLou, extensions include
    using the keys from genre song registry to provide TaggedDocuments by track
    TaggerTed is like MusicalMeg 2.0 - integrating song registry with per-track tags
    """
    def __init__(self, gen, lydir, artlst, treg):
        super().__init__(gen, lydir)
        if gen:
            if isinstance(gen, str):
                self.genre: str = gen
            else:
                self.genre: list = gen
        if lydir:
            if isinstance(lydir, str):
                self.folder = lydir
            elif isinstance(lydir, list):
                self.folder: list = lydir
            else:
                self.folder = LYRICDIR
        if artlst:
            if isinstance(artlst, str):
                self.artists = clean_name(rawstr=artlst)
            if isinstance(artlst, list):
                self.artists: list = []
                for art in artlst:
                    tmpname: str = clean_name(art)
                    self.artists.append(tmpname)
            elif isinstance(artlst, set):
                self.artists: list = []
                for art in iter(artlst):
                    tmpname: str = clean_name(art)
                    self.artists.append(tmpname)
        self.trax: dict = {}
        if treg:
            if isinstance(treg, list):
                for artreg in treg:
                    if isinstance(artreg, dict):
                        rootnam = list(artreg.keys())[0]
                        rootnam = rootnam[:-3]
                        self.trax[rootnam] = artreg
            if isinstance(treg, dict):
                self.trax = treg

        self.word_artists: dict = {}
        self.artist_words: dict = {}
        self.words_trak: list = []
        self.tf_by_artist: dict = {}
        self.tf_by_trak: dict = {}
        self.tfidf_artist: dict = {}
        self.calc_artist_words()

    def __iter__(self):

        trak_ct = 0
        agg_words: int = 0
        cur_artist: str = ""
        art_taglst: list = []
        artist_feed: list = []
        if self.artists:
            for artist in self.artists:
                tmp: str = str(artist).replace(" ", "").replace(".", "")
                art_lyrf = str(self.genre) + "_" + tmp + ".lyr"
                artist_feed.append(art_lyrf)

        for lyrin, artin in feed_specific_artists(artlst=artist_feed, srcdir=self.folder):
            # lyric_cln: list = [self.lemma.lemmatize(w) for w in lyrs]
            trak_wrds: int = len(lyrin)
            if not cur_artist == artist:
                if not cur_artist == "":
                    print("    streamed %d words from %d tracks for %s" % (agg_words, trak_ct, cur_artist))
                cur_artist = artist
                trak_ct = 0
                art_taglst = []
                for trakdct in self.trax:
                    if isinstance(trakdct, dict):
                        for keytag, traknam in trakdct.items():
                            if str(keytag).startswith(cur_artist):
                                art_taglst = list(trakdct.keys())
                                break
            if art_taglst:
                try:
                    doctag = art_taglst[trak_ct]
                except IndexError:
                    doctag = cur_artist + str(trak_ct).rjust(3, "0")
                    print("Index out-of-range for registry: %s" % doctag)
            else:
                print("custom tag needed %s trak %d" % (artist, trak_ct))
                doctag = cur_artist + str(trak_ct).rjust(3, "0")

            yield TaggedDocument(words=lyrin, tags=[doctag])
            agg_words += trak_wrds
            trak_ct += 1

        return

    def calc_artist_words(self):
        """
        create a dictionary of words and their frequency, configured to be called once
        also parses any words with a count of less than 2
        :return:
        """
        for artst in self.artists:
            wrddct = self.tf_by_artist.get(artst)
            if wrddct:
                print("ERROR: calc_artist_wfreq: already has %d words" % len(wrddct))
                continue
            else:
                wrddct: dict = {}
                artstwords: set = set()

            art_wrds_tot: int = 0
            tagct: int = 0
            if isinstance(self.genre, str):
                prefix: str = self.genre
            else:
                prefix: str = self.genre[0]

            art_lf: str = prefix + "_" + artst + ".lyr"
            for lyrs, art in feed_specific_artists(artlst=art_lf, srcdir=self.folder):
                atag = artst + str(tagct).rjust(3, "0")
                trak_set: set = set(lyrs)
                trak_uniq: int = len(trak_set)
                art_wrds_tot += trak_uniq
                trak_tmp: dict = {}

                for wrd in lyrs:
                    # trak_tmp gets term frequency per track
                    if trak_tmp.get(wrd):
                        trak_tmp[wrd] += 1
                    else:
                        trak_tmp[wrd] = 1
                    # wrddct counts frequency of word at artist level
                    if wrddct.get(wrd):
                        wrddct[wrd] += 1
                    else:
                        wrddct[wrd] = 1
                    # artstwords is simply existence of word with artist
                    if wrd not in artstwords:
                        artstwords.add(wrd)

                trak_tf: dict = {}
                for wrd, ct in trak_tmp.items():
                    trak_tf[wrd] = round(ct / trak_uniq, ndigits=4)

                # track-level wrap up
                self.words_trak.append(trak_set)
                self.tf_by_trak[atag] = trak_tf
                tagct += 1

            # artist-level wrap up
            artst_tf: dict = {}
            for wrd, ct in wrddct.items():
                artst_tf[wrd] = ct / art_wrds_tot
            self.tf_by_artist[artst] = artst_tf
            print("artist_words: added %d words for artist %s" % (art_wrds_tot, artst))
            # get set of all words used by artist, and number of artist using word
            self.artist_words[artst] = artstwords
            if self.word_artists:
                if artstwords:
                    for wrd in artstwords:
                        if self.word_artists.get(wrd):
                            self.word_artists[wrd] += 1
                        else:
                            self.word_artists[wrd] = 1
            else:
                if artstwords:
                    for wrd in artstwords:
                        self.word_artists[wrd] = 1
            # end of artist feed resets: word dict per artist, unique words per artist

        return

    def calc_tfidf_by_artist(self, artist: str):
        """
        uses self.word_by_trak which gives occurrence of a word for each track in lyric
        corpus.  by iterating through it, can identify a word's 'document' frequency
        :return:
        """
        art_tfidf: dict = {}
        print("TaggerTed calc_tfidf_by_artist is running...")
        if isinstance(artist, str):
            artist_cln = artist.replace(" ", "").replace(".", "")
            if artist_cln in self.artists:
                wrddct: dict = self.tf_by_artist.get(artist_cln)
                if wrddct:
                    trax_tot: int = len(self.words_trak)
                    print("\n using %d tracks to calculate tf idf for artist %s" % (trax_tot, artist))
                    for k, v in wrddct.items():
                        trakct: int = 0
                        for trakwrds in self.words_trak:
                            if k in trakwrds:
                                trakct += 1
                        idf: float = round(log(trax_tot / trakct), ndigits=4)
                        art_tfidf[k] = round((v * idf), ndigits=4)

                self.tfidf_artist[artist_cln] = art_tfidf
                return art_tfidf
            else:
                print("%s not in self.artists, not part of this object" % artist)
        return None

class SinginSue(TaggerTed):
    """
    SinginSue allows me to inherit from TaggerTed and implement a filtered stream
    to generate-iterate lyrics.  The __iter__ method in this class will check the
    tf idf for the relevant artist, and if the word is below a threshold value
    it will be skipped/removed from the tokenized stream
    """
    def __init__(self, gen, lydir, artlst, treg):
        super().__init__(gen, lydir, artlst, treg)
        self.idf_words: dict = {}
        self.tfidf_trak: dict = {}
        self.tfidf_cutoff_pct: int = 30
        self.calc_idf_for_words()
        self.calc_tfidf_by_trak()
        self.stream_td: bool = True

    def __iter__(self):
        if self.artists:
            for artist in self.artists:
                trak_ct: int = 0
                lookups: int = 0
                artst_words: int = 0
                tmp: str = str(artist).replace(" ", "").replace(".", "")
                art_lyrfile: str = str(self.genre) + "_" + tmp + ".lyr"
                if self.tfidf_artist.get(artist):
                    art_tfidf: dict = self.tfidf_artist.get(artist)
                    art_tfivals: list = list(art_tfidf.values())
                    tfi_cutoff: float = percentile(art_tfivals, self.tfidf_cutoff_pct)
                else:
                    tfi_cutoff: float = 0.0
                if self.trax.get(artist):
                    traxdict = self.trax.get(artist)
                    art_taglst = list(traxdict.keys())
                else:
                    art_taglst = []
                for lyrs, artst in feed_specific_artists(artlst=art_lyrfile, srcdir=self.folder):
                    # lyric_cln: list = [self.lemma.lemmatize(w) for w in lyrs]
                    if art_taglst:
                        try:
                            doctag = art_taglst[trak_ct]
                        except IndexError:
                            doctag = artist + str(trak_ct).rjust(3, "0")
                            print("Index out-of-range for registry: %s" % doctag)
                    else:
                        print("custom tag needed %s trak %d" % (artist, trak_ct))
                        doctag = artist + str(trak_ct).rjust(3, "0")
                    lyrics_tok: list = []
                    # this is using count of target vs unique words for tf calc
                    tmp_tok: set = set(lyrs)
                    trak_len = len(tmp_tok)
                    for wrd in lyrs:
                        if self.idf_words.get(wrd):
                            lookups += 1
                            wrd_idf: float = self.idf_words[wrd]
                        else:
                            wrd_idf: float = self.idf_mean()
                        thistfi: float = (int(lyrs.count(wrd)) / trak_len) * wrd_idf
                        if thistfi > tfi_cutoff:
                            lyrics_tok.append(wrd)

                    if lyrics_tok:
                        if self.stream_td:
                            yield TaggedDocument(words=lyrics_tok, tags=[doctag])
                        else:
                            yield lyrics_tok

                    artst_words += len(lyrics_tok)
                    trak_ct += 1

                print("  %d words from %d tracks, %d idf filters, for %s" % (artst_words, trak_ct, lookups, artist))

        return

    def idf_mean(self):
        """
        dynamic way to return mean of idf value for all words for object
        """
        tmplst: list = list(self.idf_words.values())
        return float(mean(tmplst))

    def calc_idf_for_words(self):
        """
        go through self.word_by_trak and add each word encountered and how many tracks
        in which it appears.
        :return:
        """
        traks_for_wrd: dict = {}
        numerator: int = len(self.words_trak)
        for trak in self.words_trak:
            for wrd in iter(trak):
                if traks_for_wrd.get(wrd):
                    traks_for_wrd[wrd] += 1
                else:
                    traks_for_wrd[wrd] = 1

        word_count: int = len(traks_for_wrd)
        for wrd, ct in traks_for_wrd.items():
            self.idf_words[wrd] = round(log(numerator / ct), ndigits=4)
        print("filled idf_words for %d words in Object" % word_count)

        return

    def calc_tfidf_by_trak(self):
        """
        uses self.tf_by_trak - each word has value of trak occurrences / trak words
        and uses self.word_by_trak - each trak has set for word's existence, must
            iterate thru and sum for word exists
        :return:
        """
        print("  -- PATIENCE: creating tf idf by track for all words in corpus --")
        comp_artst: int = 0
        comp_trax: int = 0
        for artist in self.artists:
            if self.trax.get(artist):
                art_trax: int = len(self.trax.get(artist))
                trax_tot: int = len(self.words_trak)
                # iterate through artists tracks
                for trak_ct in range(art_trax):
                    buildkey: str = artist + str(trak_ct).rjust(3, "0")
                    if self.tf_by_trak.get(buildkey):
                        traktf: dict = self.tf_by_trak[buildkey]
                        tmpdct: dict = {}
                        for wrd, wrdtf in traktf.items():
                            if self.idf_words.get(wrd):
                                idf: float = self.idf_words[wrd]
                            else:
                                idf: float = self.idf_mean()
                            tmpdct[wrd] = round((wrdtf * idf), ndigits=4)
                        self.tfidf_trak[buildkey] = tmpdct

                    comp_trax += 1

            comp_artst += 1
            print("tfidf by track completed for %s" % artist)

        print("tf idf by track completed for %d artists, %d tracks" % (comp_artst, comp_trax))
        return

    def calc_tfidf_by_artist(self):
        """
        average per-track tfidf for each artist word, determine a threshold value that
        can be applied as STOP words for removal.
        :return:
        """
        for artist in self.artists:
            if self.trax.get(artist):
                corp_tot: int = len(self.tf_by_trak)
                art_tot: int = len(self.trax.get(artist))
                default_idf: float = log(corp_tot / art_tot)
                artist_tfidf: dict = {}
                artist_tfs: dict = {}
                idf_valid: int = 0
                for i in range(art_tot):
                    buildkey = artist + str(i).rjust(3, "0")
                    # get word tf for each track, multiply by word idf
                    if self.tf_by_trak.get(buildkey):
                        tfwrds: dict = self.tf_by_trak[buildkey]
                        for k, v in tfwrds.items():
                            if artist_tfs.get(k):
                                tmplst: list = artist_tfs[k]
                                tmplst.append(v)
                            else:
                                tmplst: list = [v]
                            artist_tfs[k] = tmplst

                for wrd, vals in artist_tfs.items():
                    val_avg: float = round(mean(vals), ndigits=4)
                    if self.idf_words.get(wrd):
                        artist_tfidf[wrd] = round(val_avg * self.idf_words[wrd], ndigits=4)
                        idf_valid += 1
                    else:
                        artist_tfidf[wrd] = round(val_avg * default_idf, ndigits=4)

                self.tfidf_artist[artist] = artist_tfidf
                print("tfidf_artist for %s, %d words used idf_words" % (artist, idf_valid))

        return

class MusicalMeg(SinginSue):
    """
    new musical class which uses Genre Aggregator for instantiation data.
    """
    def __init__(self, ga_obj: Genre_Aggregator, lydir: str=LYRICDIR):

        self.folder = lydir
        if isinstance(ga_obj, Genre_Aggregator):
            self.genre = ga_obj.genre
            self.trax = ga_obj.trax
            self.artists = ga_obj.artist_list
        self.words_raw_count: int = 0
        self.word_freq: dict = {}
        self.corpbow = []
        self.corpdic: Dictionary = {}
        self.word_artists: dict = {}
        self.artist_words: dict = {}
        self.words_trak: list = []
        self.idf_words: dict = {}
        self.tf_by_artist: dict = {}
        self.tf_by_trak: dict = {}
        self.tfidf_artist: dict = {}
        self.tfidf_trak: dict = {}
        self.tfidf_cutoff_pct: int = 30
        self.calc_word_freq()
        self.calc_artist_words()
        self.calc_idf_for_words()
        self.stream_td: bool = True

    def __iter__(self):
        if self.artists:
            for artist in self.artists:
                trak_ct: int = 0
                lookups: int = 0
                artst_words: int = 0
                tmp: str = str(artist).replace(" ", "").replace(".", "")
                art_lyrfile: str = str(self.genre) + "_" + tmp + ".lyr"
                if self.tfidf_artist.get(artist):
                    art_tfidf: dict = self.tfidf_artist.get(artist)
                    art_tfivals: list = list(art_tfidf.values())
                    tfi_cutoff: float = percentile(art_tfivals, self.tfidf_cutoff_pct)
                else:
                    tfi_cutoff: float = 0.0
                if self.trax.get(artist):
                    traxdict = self.trax.get(artist)
                    art_taglst = list(traxdict.keys())
                else:
                    art_taglst = []
                for lyrs, artst in feed_specific_artists(artlst=art_lyrfile, srcdir=self.folder):
                    # lyric_cln: list = [self.lemma.lemmatize(w) for w in lyrs]
                    if art_taglst:
                        try:
                            doctag = art_taglst[trak_ct]
                        except IndexError:
                            doctag = artist + str(trak_ct).rjust(3, "0")
                            print("Index out-of-range for registry: %s" % doctag)
                    else:
                        print("custom tag needed %s trak %d" % (artist, trak_ct))
                        doctag = artist + str(trak_ct).rjust(3, "0")
                    lyrics_tok: list = []
                    # this is using count of target vs unique words for tf calc
                    tmp_tok: set = set(lyrs)
                    trak_len = len(tmp_tok)
                    for wrd in lyrs:
                        if self.idf_words.get(wrd):
                            lookups += 1
                            wrd_idf: float = self.idf_words[wrd]
                        else:
                            wrd_idf: float = self.idf_mean()
                        thistfi: float = (int(lyrs.count(wrd)) / trak_len) * wrd_idf
                        if thistfi > tfi_cutoff:
                            lyrics_tok.append(wrd)

                    if lyrics_tok:
                        if self.stream_td:
                            yield TaggedDocument(words=lyrics_tok, tags=[doctag])
                        else:
                            # for test data, can keep tag as source trace
                            yield lyrics_tok, doctag

                    artst_words += len(lyrics_tok)
                    trak_ct += 1

                print("  %d words from %d tracks, %d idf filters, for %s" % (artst_words, trak_ct, lookups, artist))

        return

    def calc_word_freq(self, min_count: int = 3):
        """
        create a dictionary of words and their frequency, configured to be called once
        also parses any words with a count of less than 2
        :param min_count: only include words that occur at least this many times
        :return:
        """
        self.words_raw_count = 0
        cur_artist: str = ""
        artist_ct: int = 0
        for lyrics, artist, eot in feed_sent(self.folder, prefix=self.genre, eot=False):
            if not cur_artist.startswith(artist):
                if not cur_artist == "":
                    print("calc_word_freq: %d artists, %d words raw count" %
                          (artist_ct, self.words_raw_count))
                artist_ct += 1
                cur_artist = artist
            for wrd in lyrics:
                self.words_raw_count += 1
                if wrd not in self.word_freq:
                    # unique (new) words...
                    self.word_freq[wrd] = 1
                else:
                    # repeat words...
                    self.word_freq[wrd] += 1

        self.word_freq = {k: v for k, v in self.word_freq.items() if v >= min_count}
        self.word_freq = {k: v for k, v in sorted(self.word_freq.items(),
                                                  key=lambda item: item[1], reverse=True)}

        return

    def calc_artist_words(self):
        """
        create a dictionary of words and their frequency, configured to be called once
        also parses any words with a count of less than 2
        :return:
        """
        for artst in self.artists:
            wrddct = self.tf_by_artist.get(artst)
            if wrddct:
                print("ERROR: calc_artist_wfreq: already has %d words" % len(wrddct))
                continue
            else:
                wrddct: dict = {}
                artstwords: set = set()

            art_wrds_tot: int = 0
            tagct: int = 0
            if isinstance(self.genre, str):
                prefix: str = self.genre
            else:
                prefix: str = self.genre[0]

            art_lf: str = prefix + "_" + artst + ".lyr"
            for lyrs, art in feed_specific_artists(artlst=art_lf, srcdir=self.folder):
                atag = artst + str(tagct).rjust(3, "0")
                trak_set: set = set(lyrs)
                trak_uniq: int = len(trak_set)
                art_wrds_tot += trak_uniq
                trak_tmp: dict = {}

                for wrd in lyrs:
                    # trak_tmp gets term frequency per track
                    if trak_tmp.get(wrd):
                        trak_tmp[wrd] += 1
                    else:
                        trak_tmp[wrd] = 1
                    # wrddct counts frequency of word at artist level
                    if wrddct.get(wrd):
                        wrddct[wrd] += 1
                    else:
                        wrddct[wrd] = 1
                    # artstwords is simply existence of word with artist
                    if wrd not in artstwords:
                        artstwords.add(wrd)

                trak_tf: dict = {}
                for wrd, ct in trak_tmp.items():
                    trak_tf[wrd] = round(ct / trak_uniq, ndigits=4)

                # track-level wrap up
                self.words_trak.append(trak_set)
                self.tf_by_trak[atag] = trak_tf
                tagct += 1

            # artist-level wrap up
            artst_tf: dict = {}
            for wrd, ct in wrddct.items():
                artst_tf[wrd] = ct / art_wrds_tot
            self.tf_by_artist[artst] = artst_tf
            print("artist_words: added %d words for artist %s" % (art_wrds_tot, artst))
            # get set of all words used by artist, and number of artist using word
            self.artist_words[artst] = artstwords
            if self.word_artists:
                if artstwords:
                    for wrd in artstwords:
                        if self.word_artists.get(wrd):
                            self.word_artists[wrd] += 1
                        else:
                            self.word_artists[wrd] = 1
            else:
                if artstwords:
                    for wrd in artstwords:
                        self.word_artists[wrd] = 1
            # end of artist feed resets: word dict per artist, unique words per artist

        return

    def calc_idf_for_words(self):
        """
        go through self.word_by_trak and add each word encountered and how many tracks
        in which it appears.
        :return:
        """
        traks_for_wrd: dict = {}
        numerator: int = len(self.words_trak)
        for trak in self.words_trak:
            for wrd in iter(trak):
                if traks_for_wrd.get(wrd):
                    traks_for_wrd[wrd] += 1
                else:
                    traks_for_wrd[wrd] = 1

        word_count: int = len(traks_for_wrd)
        for wrd, ct in traks_for_wrd.items():
            self.idf_words[wrd] = round(log(numerator / ct), ndigits=4)
        print("filled idf_words for %d words in Object" % word_count)

        return

    def idf_mean(self):
        """
        dynamic way to return mean of idf value for all words for object
        """
        tmplst: list = list(self.idf_words.values())
        return float(mean(tmplst))

    def calc_tfidf_by_trak(self):
        """
        uses self.tf_by_trak - each word has value of trak occurrences / trak words
        and uses self.word_by_trak - each trak has set for word's existence, must
            iterate thru and sum for word exists
        :return:
        """
        print("  -- PATIENCE: creating tf idf by track for all words in corpus --")
        comp_artst: int = 0
        comp_trax: int = 0
        for artist in self.artists:
            if self.trax.get(artist):
                art_trax: int = len(self.trax.get(artist))
                trax_tot: int = len(self.words_trak)
                # iterate through artists tracks
                for trak_ct in range(art_trax):
                    buildkey: str = artist + str(trak_ct).rjust(3, "0")
                    if self.tf_by_trak.get(buildkey):
                        traktf: dict = self.tf_by_trak[buildkey]
                        tmpdct: dict = {}
                        for wrd, wrdtf in traktf.items():
                            if self.idf_words.get(wrd):
                                idf: float = self.idf_words[wrd]
                            else:
                                idf: float = self.idf_mean()
                            tmpdct[wrd] = round((wrdtf * idf), ndigits=4)
                        self.tfidf_trak[buildkey] = tmpdct

                    comp_trax += 1

            comp_artst += 1
            print("tfidf by track completed for %s" % artist)

        print("tf idf by track completed for %d artists, %d tracks" % (comp_artst, comp_trax))
        return

    def calc_tfidf_by_artist(self):
        """
        average per-track tfidf for each artist word, determine a threshold value that
        can be applied as STOP words for removal.
        :return:
        """
        for artist in self.artists:
            if self.trax.get(artist):
                corp_tot: int = len(self.tf_by_trak)
                art_tot: int = len(self.trax.get(artist))
                default_idf: float = log(corp_tot / art_tot)
                artist_tfidf: dict = {}
                artist_tfs: dict = {}
                idf_valid: int = 0
                for i in range(art_tot):
                    buildkey = artist + str(i).rjust(3, "0")
                    # get word tf for each track, multiply by word idf
                    if self.tf_by_trak.get(buildkey):
                        tfwrds: dict = self.tf_by_trak[buildkey]
                        for k, v in tfwrds.items():
                            if artist_tfs.get(k):
                                tmplst: list = artist_tfs[k]
                                tmplst.append(v)
                            else:
                                tmplst: list = [v]
                            artist_tfs[k] = tmplst

                for wrd, vals in artist_tfs.items():
                    val_avg: float = round(mean(vals), ndigits=4)
                    if self.idf_words.get(wrd):
                        artist_tfidf[wrd] = round(val_avg * self.idf_words[wrd], ndigits=4)
                        idf_valid += 1
                    else:
                        artist_tfidf[wrd] = round(val_avg * default_idf, ndigits=4)

                self.tfidf_artist[artist] = artist_tfidf
                print("tfidf_artist for %s, %d words used idf_words" % (artist, idf_valid))

        return
