# encoding=utf-8
"""
gsSentiment is the main script for lyrics processing
"""
import nltk.data
nltk.data.path.append('/Users/bgh/dev/NLP/nltk_data')
nltk.download('stopwords', download_dir="/Users/bgh/dev/NLP/nltk_data")
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon', download_dir="/Users/bgh/dev/NLP/nltk_data")
# vsi = SentimentIntensityAnalyzer()
import gs_utils as gsutil
import gs_Lyrics as gsly
from gs_datadict import *
import topic_modeling as tpmd
import gsGenSim as gsm
import lyricsgenius as lyge
import requests

get_dead_corpus: bool = False
do_lda: bool = False
do_similarities: bool = False
save_models: bool = False

RAWDIR: str = "/Users/bgh/dev/pydev/gsSentiment/rawdata/"
albumf: str = RAWDIR + "albums.dat"
lyricsf: str = RAWDIR + "lyrics.dat"
gen_token = 'BvCw49ueZbum5ONdoJRpAtU8yY2ItAvpiVYxNjM0ky1_6TDt1GtpdivSIL2SGgV6'
client_secret = 'nwReOeyTxijV-wuOppG0M_37SroZQEToZOTpUD4F2e5Fw6YHzsqpI8kQcVE0kkCGZUuZOwsCoalbgqICD9heMA'
artist_lyrics: str = ""
artist_dict = {}
track_dict = {}
artist_titles = {}

if get_dead_corpus:
    band_id = 21900
    dead_corpf = RAWDIR + "dead_corpus.txt"
    dead_albums = {105519: "Workingman's Dead", 105547: "In the Dark",
                   105860: "Dead Set", 105546: "Reckoning", 105548: "Go to Heaven",
                   26596: "Shakedown Street", 25528: "Terrapin Station",
                   41113: "Steal Your Face", 105530: "Blues for Allah",
                   105527: "From the Mars Hotel", 105544: "Wake of the Flood",
                   105526: "History of the Grateful Dead (Bear's Choice)",
                   48231: "Europe '72", 26446: "Skull and Roses",
                   18482: "American Beauty",
                   18650: "Aoxomoxoa", 105514: "Live / Dead",
                   18576: "Anthem of the Sun", 18484: "The Grateful Dead",
                   588045: "The Very Best of Grateful Dead",
                   663866: "Dave's Picks Volume 29", 508738: "Red Rocks Amphitheatre 1978",
                   508706: "The Best of the Grateful Dead"
                   }
    lg_this = lyge.Genius(gen_token)
    the_Dead = lg_this.artist(21900)
    track_dict: dict = gsly.get_albumtrax(album_dict)
    tracks_clean: dict = get_cleantracks(track_dict)
    for album, tracks in iter(tracks_clean.items()):
        artist_lyrics, artist_titles = do_album_lyrics(album, tracks)

    if lamar_corpus:
        # api.genius.com/search?q=Kendrick%20Lamar
        "https://genius.com/artists/Kendrick-lamar"
        lamar_id = 48179
        # get albums for artist
        # Artist’s album lists can be retrieved via GET /artists/:id/albums
        songlst = lyge.api.API.artist_songs(artist_id=48179,per_page=50)
        # Album’s tracklists via GET /albums/:id/tracks
        # good kid, m.A.A.d city    album_id:  491147  Released October 22, 2012
        # do lg_this.album_tracks(491157) and get a dict back of detailed song info
        #
        trax_dct = {"swimming_pools": 81159, }

if do_lda:
    corpus = gsly.load_lyrics(dead_corpf)
    corp_clean = tpmd.prep_for_lda(corpus)
    wrd_tok = gsutil.do_wrd_tok(corp_clean)
    dtmatrix, dterms = tpmd.gensim_doc_terms(corp_clean)
    lda_mdl = tpmd.run_lda_model(dtmatrix, dterms, topics=20, train_iter=100, word_topics=True)

if do_similarities:
    gsm.get_word_similarity( , "blah", "bleeh")
    skipg_model.most_similar("rain")

if save_models:
    save_lyrics(artist_lyrics, lyricsf)
