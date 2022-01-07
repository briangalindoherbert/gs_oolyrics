# encoding=utf-8
"""
gsLyrics is the main script to build lyrics corpora and apply vector models to them
"""
import nltk.data
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from lyricsgenius import Genius

from gs_datadict import LYRICDIR, OUTDIR, NLTKDIR, ARTISTDIR, CHAR_CONVERSION,\
    QUOTEDASH_TABLE, RAP_SENTENCES, FIRSTW_SENTENCES, COUNTRY_SENTENCES, CHRISTMAS_SENTENCES
from gs_genius_wrappers import load_art_lyrics, get_artist_info, get_artist_albums, get_lyrics
from gs_genre_class import Genre_Aggregator
from gs_utils import filter_for_end_of_track, append_genre_registry, save_artist_info,\
    retrieve_genre_registry, get_artists_in_genre, do_cloud, check_outliers, plot_artist_tfidf, plot_artist_tf
from gsGenSim import do_skip_gram, document_skipgram, get_trak_lyrics, line_cleanup,\
    test_sentence_doc2vec, TaggerTed, MusicalMeg
from topic_modeling import get_gdict_and_doc2bow, run_lda_model, get_top_terms, \
    get_lda_model_topictable, print_lda_topics

nltk.data.path.append(NLTKDIR)

# boolean gate control is spread thin - need better appraoch to control what is run
build_corpus: bool = False

registry_instances: bool = True
create_lyrics_objects: bool = False
plot_tfidf: bool = False
get_in_memory_streams: bool = False

doc_models: bool = False
doc_testcases: bool = False
doc_extended_tests: bool = False
word_model: bool = False
phrases_train_test: bool = False
phrases_test2: bool = False
demo_word_model: bool = False
do_wordcloud: bool = False

lda_model: bool = False
lda_rap: bool = False
lda_country: bool = False
fasttext_model: bool = False

# useful app vars: list of all artists in each genre in corpus, string clean conversion table
genre_artists = get_artists_in_genre(incl_genres='all')
char_strip_table = str.maketrans({" ": None, "/": "-", ".": "", "'": "", "–": "-"})
unicode_translate = str.maketrans(CHAR_CONVERSION)

if build_corpus:
    corpus_adds: list = ['REM']
    genre = "firstwave"
    print("STARTING build_corpus process - %d artists to add" % len(corpus_adds))
    # call get_artist_info with an artist id or a name casts a broader search
    lg_token = '8NdJgZlN39TGL0inTrLO2R0vlaQdKrnDhjTezJ5SArugmA9cCHRKKzm0RUOiknh9'
    lg_x = Genius(access_token=lg_token, timeout=11, sleep_time=0.3,
                  remove_section_headers=True, response_format='dom', retries=2)
    get_artist_fail: list = []
    for artistx in corpus_adds:
        art_dct = get_artist_info(lg_x, namestr=artistx)
        if not art_dct:
            # keep a list of fails, usually need to adjust spelling
            get_artist_fail.append(artistx)
            continue
        else:
            art_dct['genre'] = genre
            art_dct['name'] = str(art_dct['name']).translate(unicode_translate)
            art_dct['name_cln'] = art_dct['name'].translate(char_strip_table)
            art_dct['lyr_file'] = genre + "_" + art_dct['name_cln'] + ".lyr"

            alb_dct: list = get_artist_albums(l_g=lg_x, artst=art_dct, albmax=36)
            artist_trax = get_lyrics(song_d=alb_dct, l_g=lg_x, genre=genre)
            # wrap-up: add artist's tracks to registry, standardize end-of-track markers + remove blanks
            filter_for_end_of_track(lyf=art_dct['lyr_file'], ldir=OUTDIR)
            append_genre_registry(albdct=artist_trax, genre=genre, outd=ARTISTDIR)
            save_artist_info(artdct=art_dct, genre=genre, outd=ARTISTDIR)
            print("    added %s to corpus" % art_dct['name'])
    print("    -- DONE adding to corpus --\n")

if registry_instances:
    """
    I have built multiple methods and objects to navigate lyrics files and provide them
    for unsupervised training. most allow specifying a genre or genres to include:
        1. clean_lyrics_file reads one file, cleans text, and writes to a new file
            use this if you want to feed files to embedding-training 
        2. load_lyrics reads lyrics files, cleans text, and returns py list in memory
            memory-intensive but fast approach to feeding lyrics to embedding-training
        3. LyricalLou class- create generator-iterator object to stream lyrics
            + optimal, clean approach leverages python strengths, 
            + versatile OO repository for genre, artist, Vocabulary, BoW information
        4. I defined 7 genres, they are country, metal, punk, rap, rock, firstwave, alternative.

    """
    print("\n      -- getting TRACK REGISTRIES for GENRES--\n")
    print("    registry provides sequential tags for artist tracks in lyr file")
    rap_trax = retrieve_genre_registry(genre='rap', indir=ARTISTDIR)
    firstwav_trax = retrieve_genre_registry(genre='firstwave', indir=ARTISTDIR)
    country_trax = retrieve_genre_registry(genre='country', indir=ARTISTDIR)
    alt_trax = retrieve_genre_registry(genre='alternative', indir=ARTISTDIR)

if create_lyrics_objects:
    print("\n    -- creating genre base objects --")
    print("      Genre_Aggregator instances link reference data for a music genre \n")
    rap_core = Genre_Aggregator(gen='rap')
    firstw_core = Genre_Aggregator(gen='firstwave')
    country_core = Genre_Aggregator(gen='country')
    alt_core = Genre_Aggregator(gen='alternative')

    print("\n    -- creating instances of MusicalMeg lyrics objects --")
    mm_rap = MusicalMeg(ga_obj=rap_core)
    mm_rap.calc_tfidf_by_trak()
    mm_rap.calc_tfidf_by_artist()

    mm_firstw = MusicalMeg(ga_obj=firstw_core)
    mm_firstw.calc_tfidf_by_trak()
    mm_firstw.calc_tfidf_by_artist()

    mm_country = MusicalMeg(ga_obj=country_core)
    mm_country.calc_tfidf_by_trak()
    mm_country.calc_tfidf_by_artist()

    mm_alt = MusicalMeg(ga_obj=alt_core)
    mm_alt.calc_tfidf_by_trak()
    mm_alt.calc_tfidf_by_artist()

    if plot_tfidf:
        print("    -- removing outliers and plotting tfidf for artists --")
        arts, artvals = check_outliers(mm_rap)
        for x in range(len(mm_rap.artists)):
            # plot_artist_tf(lobj=mm_rap, artst=mm_rap.artists[x])
            plot_artist_tfidf(lobj=mm_rap, artst=mm_rap.artists[x])

        arts, artvals = check_outliers(mm_firstw)
        plot_artist_tf(lobj=mm_firstw, artst=firstw_core.artist_list[8])
        plot_artist_tfidf(lobj=mm_firstw, artst=firstw_core.artist_list[8])
        plot_artist_tfidf(lobj=mm_firstw, artst=firstw_core.artist_list[6])
        plot_artist_tfidf(lobj=mm_firstw, artst=firstw_core.artist_list[1])

        arts, artvals = check_outliers(mm_alt)
        plot_artist_tfidf(lobj=mm_alt, artst=alt_core.artist_list[8])

        arts, artvals = check_outliers(mm_country)
        plot_artist_tfidf(lobj=mm_country, artst=mm_country.artists[5])

    if get_in_memory_streams:
        # flatten lyrics for single track = [item for sublist in t for item in sublist]
        trak_lyr = dict(get_trak_lyrics(dtag='GarthBrooks014', genre='country'))['lyrics']
        trak_lyr = [wrd for subl in trak_lyr for wrd in subl]

        """
        to stream word-token list NOT TaggedDocument - set <inst>.stream_td = False
        
        plot_artist_tf(lobj=mm_rap, artst=rap_artist_list[0])
        plot_artist_tfidf(lobj=mm_rap, artst=rap_artist_list[0])
    
        train_corpus: list = []
        test_corpus: list = []
        for i, td in zip(range(5200), mm_rap):
            if (i % 5) != 0:
                train_corpus.append(td)
        mm_rap.stream_td = False
        for i, td in zip(range(5200), mm_rap):
            if (i % 5) == 0:
                test_corpus.append(td)
        mm_rap.stream_td = True
        """

if doc_models:
    print("  --starting topic modeling for music tracks by artists--")

    # doc_firstw = document_skipgram(tt_firstw, passes=12, grpsize=10, dim=100, thrds=4, dm=0)
    d2v_firstw = document_skipgram(mm_firstw, passes=12, grpsize=12, dim=100, thrds=4, dm=0)
    # additional training
    d2v_firstw.train(corpus_iterable=mm_firstw, total_examples=d2v_firstw.corpus_count,
                       epochs=4, start_alpha=0.004, end_alpha=0.002)

    d2v_rap = document_skipgram(mm_rap, passes=12, grpsize=10, dim=100, thrds=4, dm=0)
    d2v_rap.train(corpus_iterable=mm_rap, total_examples=d2v_rap.corpus_count,
                       epochs=6, start_alpha=0.004, end_alpha=0.001)

    # lyrstream is list of list of TaggedDocs: artist-title for each track in corpus
    lyrstream: list = []
    for art in rap_trax:
        wrdct: int = 0
        art_td: list = []
        tag_toks: list = []
        for k, v in art.items():
            tmpv: str = str(v).translate(QUOTEDASH_TABLE)
            tmptok, wrdct = line_cleanup(txt=tmpv, word_ct=wrdct)
            # tmpv is single string with artist + track name
            tmpv = " ".join([wrd for wrd in tmptok])
            tmptok.append(tmpv)
            tag_toks.append(tmptok)
            art_td.append(TaggedDocument(words=tmptok, tags=[k]))
        lyrstream.append(art_td)

    for art_td in lyrstream:
        d2v_rap.train(corpus_iterable=art_td, total_examples=d2v_rap.corpus_count,
                        epochs=5, start_alpha=0.003, end_alpha=0.001)

    kl_lyrs = [TaggedDocument(wrds, [tags]) for wrds, tags in
               load_art_lyrics(prefix='rap', artst='KendrickLamar', artreg=rap_trax[4])]

    outk_lyrs = [TaggedDocument(wrds, [tags]) for wrds, tags in
               load_art_lyrics(prefix='rap', artst='OutKast', artreg=rap_trax[0])]

    lilj_lyrs = [TaggedDocument(wrds, [tags]) for wrds, tags in
               load_art_lyrics(prefix='rap', artst='Lil Jon', artreg=rap_trax[5])]

    d2v_rap.train(corpus_iterable=kl_lyrs, total_examples=d2v_rap.corpus_count,
                    epochs=8, start_alpha=0.02, end_alpha=0.002)

    # create models for other genres, can also split stream by train and test
    train_country: list = []
    mm_country.stream_td = False
    for i, td in zip(range(5200), mm_country):
        if (i % 20) != 0:
            train_country.append(td[0])

    test_country: list = []
    test_country_tgs: list = []
    for i, td in zip(range(5200), mm_country):
        if (i % 20) == 0:
            test_country.append(td[0])
            test_country_tgs.append(td[1])
    mm_country.stream_td = True

    d2v_country = document_skipgram(mm_country, passes=12, grpsize=10, dim=100, thrds=4, dm=0)
    d2v_country.train(corpus_iterable=train_country, total_examples=d2v_country.corpus_count,
                       epochs=8, start_alpha=0.02, end_alpha=0.002)

    print("\n    sentence tests for doc model - country genre")
    for x, line in enumerate(test_country[:17]):
        test_sentence_doc2vec(sentence=line, d2v=d2v_country, topsiz=10)

    for x, line in enumerate(test_country[13:18]):
        test_sentence_doc2vec(sentence=line, d2v=d2v_country, topsiz=10)

if doc_testcases:
    print("  -- Demonstrating Doc2Vec Model --")

    print("\n    Now test joe jackson, violent femmes on firstwave docmodel")
    for x, line in enumerate(FIRSTW_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=d2v_firstw, topsiz=10)

    # get dict of all words used in model, along with their internal index
    doc_entity = d2v_rap.dv.key_to_index
    # to get a random document ID use:
    random_doc_id = np.random.randint(low=0, high=len(d2v_rap.dv))
    print("got random doc id:")
    print(random_doc_id)
    print(d2v_rap.dv.index_to_key[random_doc_id])
    print("    -- the above can then be used to pull vectors and lyrics --")

    normed_vector = d2v_rap.dv.get_vector("KendrickLamar005", norm=True)
    doc_vector = d2v_rap.dv["KendrickLamar005"]
    doc_vectors = d2v_rap.dv.vectors

    ranks = []
    second_ranks = []
    inferred_vector = d2v_rap.infer_vector(get_trak_lyrics(dtag='KendrickLamar152', genre='rap'))
    sims = d2v_rap.dv.most_similar([inferred_vector], topn=20)
    rank = [docid for docid, sim in sims].index('KendrickLamar152')
    ranks.append(rank)
    second_ranks.append(sims[1])
    # Pick a random document from the test corpus and infer a vector from the model

    # WARNING: this will return a LARGE in-memory list
    train_corpus = [sents for i, sents in zip(range(20), mm_rap)]
    ranks = []
    second_ranks = []
    for doc_id in range(10):
        inferred_vector = d2v_rap.infer_vector(train_corpus[doc_id].words)
        sims = d2v_rap.dv.most_similar([inferred_vector], topn=len(d2v_rap.dv))

        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    # comparing previously unseen lyrics with trained lyrics models
    print("\n    testing document models with previously unseen lyrics")
    print("  lyrics excerpted from Yelawolf, Kendrick Lamar, TechN9ne, LilJon, OutKast")
    for x, line in enumerate(RAP_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=d2v_rap, topsiz=10)

    print("\n    Now test rap doc model with Country music sentences")
    for x, line in enumerate(COUNTRY_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=d2v_rap, topsiz=10)

    print("\n    Testing rap genre model with Christmas carol lyrics")
    for x, line in enumerate(CHRISTMAS_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=d2v_rap, topsiz=10)

    print("\n    sentence tests for doc model - firstwave genre")
    for x, line in enumerate(FIRSTW_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=d2v_firstw, topsiz=10)

    if doc_extended_tests:
        print("    size of vocabulary in rap genre doc model is %d\n" % len(d2v_rap.wv))
        print("    -- test for word in doc model vocabulary --")

        testword = "compton"
        if testword in d2v_rap.wv.key_to_index:
            print(f"Word {testword} appeared {d2v_rap.wv.get_vecattr(testword, 'count')} times")
        else:
            print("    testword: %s NOT in vocabulary" % testword)

        d2v_rap.wv.most_similar(positive=['adidas', 'compton'], negative=['city'], topn=5)

if do_wordcloud:
    # create stream for artist-track titles with tags, then artist lyrics too:
    # create TaggedDocument feed with artist-track-name plus sequential artist-tag
    titles_stream: list = []
    for art in rap_trax:
        wrdct: int = 0
        art_td: list = []
        tag_toks: list = []
        for k, v in art.items():
            tmpv: str = str(v).translate(QUOTEDASH_TABLE).replace("-", " ")
            tmptok, wrdct = line_cleanup(txt=tmpv, word_ct=wrdct)
            # tmpv is single string with artist + track name
            tmpv = " ".join([wrd for wrd in tmptok])
            tag_toks.append(tmpv)
        titles_stream.append(tag_toks)

    # create data from one or more artists for a wordcloud
    # can use the genre registries to identify ID of individual songs for an artist
    u2_lyrs = [wrds for wrds, tags in load_art_lyrics(prefix='alternative', artst='U2', artreg=alt_trax[9])]
    # example- created specialized list for wordcloud:
    u2_stops = ['know', 'time', 'need', 'yeah']
    # u2_tmp = u2_lyrs[106:110]  etc. - append and extend from here
    do_cloud(cloud_words=u2_lyrs, opt_stops=u2_stops, maxwrd=110)

    greenday_lyrs = [wrds for wrds, tags in load_art_lyrics(prefix='alternative', artst='GreenDay',
                                                      artreg=alt_trax[4])]

    do_cloud(cloud_words=greenday_lyrs, opt_stops=u2_stops, maxwrd=110)

    alt_stops: list = ['fuck', 'bitch', 'shit', 'lord', 'five', 'know', 'thought', 'well',
                       'yeah', 'doodoo', 'doodoodoo']
    pearljam_lyrs = [wrds for wrds, tags in load_art_lyrics(prefix='alternative',
                                                            artst='PearlJam', artreg=alt_trax[6])]
    do_cloud(cloud_words=pearljam_lyrs, opt_stops=alt_stops, maxwrd=120)

    rhcp_lyrs = [wrds for wrds, tags in
               load_art_lyrics(prefix='alternative', artst='RedHotChiliPeppers', artreg=alt_trax[5])]
    do_cloud(cloud_words=rhcp_lyrs, opt_stops=alt_stops, maxwrd=120)

    kl_lyrs = [wrds for wrds, tags in load_art_lyrics(prefix='rap', artst='LilJon', artreg=rap_trax[5])]

    lilj_lyrs = [wrds for wrds, tags in
               load_art_lyrics(prefix='rap', artst='LilJon', artreg=rap_trax[5])]
    lilj_lyrs.append(titles_stream[5])

    # to get lyrics for a track, look for track name in registry, then get_tag_source
    print(rap_trax)
    get_trak_lyrics(dtag='KendrickLamar152', genre='rap')

    # do_cloud in gs_utils can receive one or more word-tokenized lists and a stop list
    rap_cloud_stops = ['nigga', 'fuck', 'fucking', 'bitch', 'pussy', 'hoes']
    do_cloud(cloud_words=lilj_lyrs, opt_stops=rap_cloud_stops, maxwrd=120)

if word_model:
    print("  -- start unsupervised training of vectors for word embeddings --")

    """
     lyrics_fw = load_lyrics(LYRICDIR, prefix="firstwave", sep_artists=True)
     lyrlist = load_tag_docs("rap_KendrickLamar.lyr")

     LL_rap = LyricalLou(gen_prefx='rap', folder=LYRICDIR)
     LL_rap.create_bow()
     LL_1stwav = LyricalLou(gen_prefx='firstwave')
     LL_1stwav.create_bow()
     LL_country = LyricalLou(gen_prefx='country')
     LL_country.create_bow()
     LL_all = LyricalLou(gen_prefx='all')
     LL_all.create_bow()
     """
    tt_rap = TaggerTed(gen='rap', lydir=LYRICDIR, artlst=rap_core.artist_list, treg=rap_trax)
    w2v_rap = do_skip_gram(iterlyr=tt_rap, dim=200, thrds=4, passes=6, grpsz=8, maxv=42000)
    # w2v_rap = do_skip_gram(iterlyr=mm_rap, dim=200, thrds=4, passes=6, grpsz=8, maxv=42000)

    if demo_word_model:
        """
        demo section applies word embedding tools and shows off capabilities of the models
        
        I plan to modify it to demonstrate one or two specific use cases rather than
        simply demonstrating random capabilities
        """
        from sklearn.decomposition import PCA
        print("\n---- vector model demonstration section ----")

        if w2v_rap:
            w2v_rap.predict_output_word(context_words_list=["homies", "friends",
                                                            "rollin", "dirty"], topn=5)
            X = w2v_rap[w2v_rap.wv.vocab]
            pca = PCA(n_components=2)
            result = pca.fit_transform(X)
            # convert from matplot bullshit: result[:, 0], result[:, 1]
            words = list(w2v_rap.wv.vocab)
            pca_dct: dict = {}
            for i, word in enumerate(words):
                pca_dct[word] = xy=(result[i, 0], result[i, 1])

            result = w2v_rap.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
            print("classic nlp analogy: man is to woman as king is to: %s" % result)

            more_examples = ["he his she", "big bigger bad"]
            for example in more_examples:
                a, b, x = example.split()
                predicted = w2v_rap.most_similar([x, b], [a])[0][0]
                print("%s is to %s as %s is to %s" % (a, b, x, predicted))

            mismatched: list = ["breakfast", "cereal", "dinner", "lunch"]
            w2v_rap.doesnt_match(mismatched)

            w2v_rap.wv.most_similar("music")

            analogies: list = [
                {'positive': ['enemies', 'love'], 'negative': ['friends']},
                {'positive': ['friend', 'lover'], 'negative': ['fuck']},
                {'positive': ['woman', 'strong'], 'negative': ['man']}
            ]
            for example in analogies:
                if isinstance(example, dict):
                    pos: list = example.get('positive')
                    neg: list = example.get('negative')
                    print(w2v_rap.wv.most_similar(positive=pos, negative=neg, topn=3))

        # k-means clustering to see how word and phrase vectors group

if phrases_train_test:
    print("   -- Analyzing Artist Word Use and Modeling Phrases in Corpus --")
    # lets get unique language use by artists as well as phrases
    # first, lets find words which are used by less than half the genre's artists
    # rap genre has 51,072 words and 16 artists, self.word_artists tells how many use word
    rare_words: dict = {}
    artist_ct: int = len(mm_rap.artists)
    rare_ct = round(artist_ct * 0.4, ndigits=0)
    for wrd, ct in mm_rap.word_artists.items():
        if ct <= rare_ct:
            if mm_rap.word_freq.get(wrd):
                if mm_rap.word_freq.get(wrd) > 5:
                    rare_words[wrd] = mm_rap.word_freq.get(wrd)

    train_phrase_corp: list = []
    test_phrase_corp: list = []
    test_tagsource: list = []
    mm_country.stream_td = False
    for i, td in zip(range(3200), mm_country):
        if (i % 5) != 0:
            train_phrase_corp.append(td[0])
    for i, td in zip(range(3200), mm_country):
        if (i % 5) == 0:
            test_phrase_corp.append(td[0])
            test_tagsource.append(td[1])
    mm_country.stream_td = True


if phrases_test2:
    train_phrase: list = []
    test_phrase: list = []
    test_tagsource: list = []
    mm_firstw.stream_td = False
    for i, td in zip(range(len(mm_firstw.words_trak)), mm_firstw):
        if (i % 5) != 0:
            train_phrase.append(td[0])
    for i, td in zip(range(len(mm_firstw.words_trak)), mm_firstw):
        if (i % 5) == 0:
            test_phrase.append(td[0])
            test_tagsource.append(td[1])
    mm_firstw.stream_td = True

    phrase_mdl = Phrases(train_phrase, min_count=4)
    phrases: dict = {}
    for phrase, score in phrase_mdl.find_phrases(test_phrase).items():
        phrases[phrase] = score
        print(phrase, score)

if fasttext_model:
    """
    use of FastText in this app is incomplete, mainly code placeholder for now! 
    have not worked out bugs in compute_similarity method

    import fasttext

    fw_ftmdl = fasttext.train_unsupervised(fwlyrlist, model='skipgram', epoch=8, lr=0.04, thread=4)
    gs_fasttext.compute_similarity(fw_ftmdl, fwlyrlist)
    fw_ftmdl.get_nearest_neighbors("love")
    fw_ftmdl.get_nearest_neighbors("hate")
    """

if lda_model:
    # LDA assumes documents are produced from a mixture of topics
    # Topics generate words based on their probability distribution.
    print("  ---- CREATE LDA model for topic modeling ----\n")
    tot_topics = 16
    tot_words = 6

    train_rap: list = []
    test_rap: list = []
    test_tags: list = []
    mm_rap.stream_td = False
    for i, td in zip(range(5200), mm_rap):
        if (i % 10) != 0:
            tmpl: list = []
            for wrd in td[0]:
                if len(wrd) > 3:
                    tmpl.append(wrd)
            train_rap.append(tmpl)
            # train_rap_tags.append(td[1])
        else:
            tmpl: list = []
            for wrd in td[0]:
                if len(wrd) > 3:
                    tmpl.append(wrd)
            test_rap.append(tmpl)
            test_tags.append(td[1])
    mm_rap.stream_td = True

    if lda_rap:
        print("  ---- DISPLAY TEST CASES FOR LDA model ----\n")
        dtmatrix, dterms = get_gdict_and_doc2bow(train_rap, at_least_docs=80)
        # must be in at least n docs, no more than x % of docs, and keep first y most frequent
        print("length of lda Dictionary after pruning is %d" % len(dterms))
        lda_mdl = run_lda_model(dtmatrix, dterms, topics=20, train_iter=100, word_topics=True)
        lda_mdl.top_topics(corpus=dtmatrix, topn=6)
        print(lda_mdl.print_topics(num_topics=tot_topics, num_words=tot_words))
        # get topic distributions
        topic_dist = lda_mdl.state.get_lambda()
        # get topic terms
        for i, topic in lda_mdl.show_topics(formatted=True, num_topics=tot_topics, num_words=tot_words):
            print(str(i) + ": " + topic)
            print()
        topics_lda = lda_mdl.show_topics(num_words=6, log=True, formatted=True)

        # get difference between two lda models:
        # mdiff, annotation = m1.diff(m2)
        # topic_diff = mdiff  # get matrix with difference for each topic pair

        # show terms by document for docs with > 10 terms for up to 50 docs
        counter: int = 0
        for x in range(len(dtmatrix)):
            if len(dtmatrix[x]) > 12:
                counter += 1
                if counter < 50:
                    maptopic = lda_mdl[dtmatrix[x]]
                    print([dterms.id2token[maptopic[0][i][0]] for i in range(len(maptopic[0]))])

        testtok = [['he', 'brushed', 'em', 'walked', 'back', 'kentucky', 'fried', 'chicken',
                    'chicken', 'spot', 'was', 'light', 'skinned', 'nigga', 'talked', 'lot',
                    'curly', 'top', 'gap', 'teeth,', 'he', 'worked', 'window']]
        other_bow = [dterms.doc2bow(txt) for txt in testtok]
        test_doc = other_bow[0]
        # ldamdl_upd = lda_mdl.update(other_bow)

    if lda_country:
        print("\n  -- LDA creates document-topic and topic-terms matrices for a corpus --")
        #
        train_country: list = []
        test_country: list = []
        test_country_tags: list = []
        mm_country.stream_td = False
        for i, td in zip(range(5200), mm_country):
            if (i % 10) != 0:
                tmpl: list = []
                for wrd in td[0]:
                    if len(wrd) > 3:
                        tmpl.append(wrd)
                train_country.append(tmpl)
            else:
                tmpl: list = []
                for wrd in td[0]:
                    if len(wrd) > 3:
                        tmpl.append(wrd)
                test_country.append(tmpl)
                test_country_tags.append(td[1])
        mm_country.stream_td = True

        doc_term_ctry, dict_ctry = get_gdict_and_doc2bow(train_country, at_least_docs=50)
        # must be in at least n docs, no more than x % of docs, and keep first y most frequent
        print("length of lda Dictionary after pruning is %d" % len(dict_ctry))
        lda_country = run_lda_model(doc_term_ctry, dict_ctry, topics=20, train_iter=100, word_topics=True)
        lda_country.top_topics(corpus=doc_term_ctry, topn=6)
        print(lda_country.print_topics(num_topics=tot_topics, num_words=tot_words))
        ctry_topics = print_lda_topics(lda_country, ntopics=20)
