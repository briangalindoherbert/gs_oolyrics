# encoding=utf-8
"""
gsLyrics is the main script to build lyrics corpora and apply vector models to them
"""
import numpy as np
from gs_datadict import LYRICDIR, OUTDIR, NLTKDIR, ARTISTDIR, GENRES, CHAR_CONVERSION, \
    QUOTEDASH_TABLE, RAP_SENTENCES, FIRSTW_SENTENCES, COUNTRY_SENTENCES, CHRISTMAS_SENTENCES
from gs_datadict import rapa, firstwavea, alternativea, countrya
from gs_utils import filter_for_end_of_track, append_genre_registry, save_artist_info, \
    retrieve_genre_registry, get_artists_in_genre, do_cloud, compare_artist_tfidf, \
    check_outliers, plot_artist_tfidf, plot_artist_tf
from gs_genius_wrappers import load_art_lyrics, get_artist_info, get_artist_albums, get_lyrics
from topic_modeling import get_gdict_and_doc2bow, run_lda_model
from gsGenSim import do_skip_gram, document_skipgram, get_tag_source, line_cleanup, \
    test_sentence_doc2vec, TaggerTed, SinginSue, MusicalMeg
from gs_genre_class import Genre_Aggregator
from gensim.models.doc2vec import TaggedDocument
from lyricsgenius import Genius
import nltk.data

nltk.data.path.append(NLTKDIR)
build_corpus: bool = False
prep_objects: bool = True
prep_objects2: bool = False

doc_models: bool = False
doc_testcases: bool = False
doc_extended_tests: bool = False
word_model: bool = False
phrases_train_test: bool = False
phrases_test2: bool = False
demo_word_model: bool = False
prep_cloud_data: bool = False

lda_model: bool = False
lda_testcases: bool = False
fasttext_model: bool = False

# useful app vars: list of all artists in each genre in corpus, string clean conversion table
genre_artists = get_artists_in_genre(incl_genres='all')
char_strip_table = str.maketrans({" ": None, "/": "-", ".": "", "'": "", "–":"-"})
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

if prep_objects:
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
        
    after upgrading my python 3.9.7 environment, incl. upgrade of scipy, no gensim errors
    had been getting AttributeError no keyed vector for 'lockf' with Word2Vec
    """
    print("\n  ------ PREP LYRICS OBJECTS: instantiate lyrics object for streaming ------")

    print("\n      --GETTING TRACK REGISTRIES BY GENRE--\n")
    # genre registry provides tags matching artist tracks
    firstwav_trax = retrieve_genre_registry(genre='firstwave', indir=ARTISTDIR)
    country_trax = retrieve_genre_registry(genre='country', indir=ARTISTDIR)
    # metal_trax = retrieve_genre_registry(genre='metal', indir=ARTISTDIR)
    rap_trax = retrieve_genre_registry(genre='rap', indir=ARTISTDIR)

    print("\n    creating genre base objects")
    alt_trax = retrieve_genre_registry(genre='alternative', indir=ARTISTDIR)
    rap_core = Genre_Aggregator(gen='rap')
    country_core = Genre_Aggregator(gen='country')
    alt_core = Genre_Aggregator(gen='alternative')
    firstw_core = Genre_Aggregator(gen='firstwave')

    print("\n creating instances of lyrics objects")
    tt_alt = TaggerTed(gen='alternative', lydir=LYRICDIR, artlst=rapa, treg=alt_trax)
    mm_alt = MusicalMeg(ga_obj=alt_core)
    mm_alt.calc_tfidf_by_trak()
    mm_alt.calc_tfidf_by_artist()

    alt_artists: list = []
    # get artist from alt_core.artist_list attribute

    plot_artist_tf(lobj=mm_alt, artst=alt_core.artist_list[8])
    plot_artist_tfidf(lobj=mm_alt, artst=alt_core.artist_list[8])

    mm_rap = MusicalMeg(ga_obj=rap_core)
    mm_rap.calc_tfidf_by_trak()
    mm_rap.calc_tfidf_by_artist()

if prep_objects2:
    mm_country = MusicalMeg(ga_obj=country_core)
    mm_country.calc_tfidf_by_trak()
    mm_country.calc_tfidf_by_artist()
    arts, artvals = check_outliers(mm_country)

    plot_artist_tfidf(lobj=mm_country, artst=mm_country.artists[5])
    plot_artist_tf(lobj=mm_country, artst=mm_country.artists[5])

    mm_firstw = MusicalMeg(ga_obj=firstw_core)
    mm_firstw.calc_tfidf_by_trak()
    mm_firstw.calc_tfidf_by_artist()
    arts, artvals = check_outliers(mm_firstw)

    firstw_artists: list = []
    for x in range(len(mm_firstw.artists)):
        firstw_artists.append(mm_firstw.artists[x])

    plot_artist_tfidf(lobj=mm_firstw, artst=firstw_artists[16])
    plot_artist_tfidf(lobj=mm_firstw, artst=firstw_artists[6])
    plot_artist_tfidf(lobj=mm_firstw, artst=firstw_artists[1])

    # prior to the histogram plots of tfidf, good to remove outliers as below!
    arts, artvals = check_outliers(mm_rap)

    rap_artist_list: list = []
    for x in range(len(mm_rap.artists)):
        rap_artist_list.append(mm_rap.artists[x])
    plot_artist_tfidf(lobj=mm_rap, artst=rap_artist_list[3])

    country_artists: list = []
    for x in range(len(mm_country.artists)):
        country_artists.append(mm_country.artists[x])

    plot_artist_tf(lobj=mm_country, artst=country_artists[2])
    plot_artist_tfidf(lobj=mm_country, artst=country_artists[2])

    # tip to flatten list: flat_list = [item for sublist in t for item in sublist]
    trak_lyr = dict(get_tag_source(dtag='GarthBrooks014', genre='country'))['lyrics']
    trak_lyr = [wrd for subl in trak_lyr for wrd in subl]

    """
    plot_artist_tf(lobj=mm_rap, artst=rap_artist_list[0])
    plot_artist_tfidf(lobj=mm_rap, artst=rap_artist_list[0])

    tt_rap = TaggerTed(gen='rap', lydir=LYRICDIR, artlst=rapa, treg=rap_trax)
    tt_rap.calc_word_freq()
    tt_country = TaggerTed(gen='country', lydir=LYRICDIR, treg=None, artlst=countrya)
    ss_rap = SinginSue(gen='rap', lydir=LYRICDIR, artlst=rapa, treg=rap_trax)
    ss_rap.artist_tfidf_by_trak()
    ss_rap.create_artist_final_tfidf()
    ss_rap.create_bow()

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
    doc_model = document_skipgram(mm_rap, passes=12, grpsize=12, dim=100, thrds=4, dm=0)
    # additional training
    doc_model.train(corpus_iterable=mm_rap, total_examples=doc_model.corpus_count,
                       epochs=4, start_alpha=0.004, end_alpha=0.002)

    filter_model = document_skipgram(mm_rap, passes=12, grpsize=10, dim=100, thrds=4, dm=0)
    filter_model.train(corpus_iterable=mm_rap, total_examples=filter_model.corpus_count,
                       epochs=6, start_alpha=0.004, end_alpha=0.001)

    # this adds the lyrstream- TaggedDocs with name of each track in corpus
    # create TaggedDocument feed with artist-track-name plus sequential artist-tag
    td_tag_stream: list = []
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
        td_tag_stream.append(tag_toks)
        lyrstream.append(art_td)

    for art_td in lyrstream:
        doc_model.train(corpus_iterable=art_td, total_examples=doc_model.corpus_count,
                        epochs=5, start_alpha=0.003, end_alpha=0.001)

    kl_lyrs = [TaggedDocument(wrds, [tags]) for wrds, tags in
               load_art_lyrics(prefix='rap', artst='KendrickLamar', artreg=rap_trax[4])]

    outk_lyrs = [TaggedDocument(wrds, [tags]) for wrds, tags in
               load_art_lyrics(prefix='rap', artst='OutKast', artreg=rap_trax[0])]

    lilj_lyrs = [TaggedDocument(wrds, [tags]) for wrds, tags in
               load_art_lyrics(prefix='rap', artst='Lil Jon', artreg=rap_trax[5])]

    filter_model.train(corpus_iterable=kl_lyrs, total_examples=filter_model.corpus_count,
                    epochs=8, start_alpha=0.02, end_alpha=0.002)

    # create models for other genres, can also split stream by train and test
    train_country: list = []
    for i, td in zip(range(5200), mm_country):
        if (i % 20) != 0:
            train_country.append(td)

    mm_country.stream_td = False
    test_country: list = []
    for i, td in zip(range(5200), mm_country):
        if (i % 20) == 0:
            test_country.append(td)
    mm_country.stream_td = True

    country_model = document_skipgram(train_country, passes=12, grpsize=10, dim=100, thrds=4, dm=0)
    country_model.train(corpus_iterable=train_country, total_examples=country_model.corpus_count,
                       epochs=8, start_alpha=0.02, end_alpha=0.002)

    print("\n    sentence tests for doc model - country genre")
    for x, line in enumerate(test_country[:17]):
        test_sentence_doc2vec(sentence=line, d2v=country_model, topsiz=10)

    for x, line in enumerate(test_country[13:18]):
        test_sentence_doc2vec(sentence=line, d2v=country_model, topsiz=10)

if doc_testcases:
    print("  -- Demonstrating Doc2Vec Model --")

    # get dict of all words used in model, along with their internal index
    doc_entity = doc_model.dv.key_to_index
    # to get a random document ID use:
    random_doc_id = np.random.randint(low=0, high=len(doc_model.dv))
    print("got random doc id:")
    print(random_doc_id)
    print(doc_model.dv.index_to_key[random_doc_id])
    print("    -- the above can then be used to pull vectors and lyrics --")

    normed_vector = doc_model.dv.get_vector("KendrickLamar005", norm=True)
    doc_vector = doc_model.dv["KendrickLamar005"]
    doc_vectors = doc_model.dv.vectors

    ranks = []
    second_ranks = []
    inferred_vector = doc_model.infer_vector(get_tag_source(dtag='KendrickLamar152', genre='rap'))
    sims = doc_model.dv.most_similar([inferred_vector], topn=20)
    rank = [docid for docid, sim in sims].index('KendrickLamar152')
    ranks.append(rank)
    second_ranks.append(sims[1])
    # Pick a random document from the test corpus and infer a vector from the model

    # WARNING: this will return a LARGE in-memory list
    train_corpus = [sents for i, sents in zip(range(20), mm_rap)]
    ranks = []
    second_ranks = []
    for doc_id in range(10):
        inferred_vector = doc_model.infer_vector(train_corpus[doc_id].words)
        sims = doc_model.dv.most_similar([inferred_vector], topn=len(doc_model.dv))

        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    # comparing random lyrics in genre to model for similarity
    # rap_sentences are from Yelawolf, Kendrick Lamar, and TechN9ne
    print("\n    sentence tests for taggerted doc model - rap genre")
    print(" lyrics excerpted from Yelawolf, Kendrick Lamar, TechN9ne, LilJon, OutKast")
    for x, line in enumerate(RAP_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=doc_model, topsiz=10)

    print("\n    Now test singinsue doc model - rap genre on same sentences")
    for x, line in enumerate(RAP_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=filter_model, topsiz=10)

    print("\n    Now test TaggerTed doc model - rap genre on Country music sentences")
    for x, line in enumerate(COUNTRY_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=doc_model, topsiz=10)

    print("\n    And test TaggerTed doc model - rap genre on Christmas carol lyrics!")
    for x, line in enumerate(CHRISTMAS_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=doc_model, topsiz=10)

    print("\n    sentence tests for doc model - firstwave genre")
    for x, line in enumerate(FIRSTW_SENTENCES):
        test_sentence_doc2vec(sentence=line, d2v=doc_model, topsiz=10)

    if doc_extended_tests:
        print("    size of word vocabulary in doc model is %d\n" % len(doc_model.wv))
        print("    -- test for word in doc model vocabulary --")

        testword = "compton"
        if testword in doc_model.wv.key_to_index:
            print(f"Word {testword} appeared {doc_model.wv.get_vecattr(testword, 'count')} times")
        else:
            print("    testword: %s NOT in vocabulary" % testword)

        doc_model.wv.most_similar(positive=['adidas', 'compton'], negative=['city'], topn=5)

if prep_cloud_data:
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
    u2_lyrs = [wrds for wrds, tags in load_art_lyrics(prefix='alternative', artst='U2',
                                                      artreg=alt_trax[9])]
    # example- created specialized list for wordcloud:
    u2_stops = ['know', 'time', 'need', 'yeah']
    # u2_tmp = u2_lyrs[106:110]  etc. - append and extend from here

    do_cloud(batch_tw_wrds=u2_lyrs, opt_stops=u2_stops, maxwrd=110)

    alt_stops: list = ['fuck', 'bitch', 'shit', 'lord', 'five', 'know', 'thought', 'well',
                       'yeah', 'doodoo', 'doodoodoo']
    pearljam_lyrs = [wrds for wrds, tags in load_art_lyrics(prefix='alternative',
                                                            artst='PearlJam', artreg=alt_trax[6])]
    do_cloud(batch_tw_wrds=pearljam_lyrs, opt_stops=alt_stops, maxwrd=120)

    rhcp_lyrs = [wrds for wrds, tags in
               load_art_lyrics(prefix='alternative', artst='RedHotChiliPeppers', artreg=alt_trax[5])]
    do_cloud(batch_tw_wrds=rhcp_lyrs, opt_stops=alt_stops, maxwrd=120)

    kl_lyrs = [wrds for wrds, tags in
                 load_art_lyrics(prefix='rap', artst='LilJon', artreg=rap_trax[5])]

    lilj_lyrs = [wrds for wrds, tags in
               load_art_lyrics(prefix='rap', artst='LilJon', artreg=rap_trax[5])]
    lilj_lyrs.append(titles_stream[5])

    # to get lyrics for a track, look for track name in registry, then get_tag_source
    print(rap_trax)
    get_tag_source(dtag='KendrickLamar152', genre='rap')


    # do_cloud in gs_utils can receive one or more word-tokenized lists and a stop list
    rap_cloud_stops = ['nigga', 'fuck', 'fucking', 'bitch', 'pussy', 'hoes']
    do_cloud(batch_tw_wrds=lilj_lyrs, stops=rap_cloud_stops, maxwrd=120)

if word_model:
    """
    this section does unsupervised training with gensim Word2Vec,
    I plan to extend this section to also train doc2vec and phrase2vec
    """
    print("  --doing unsupervised training for word embeddings--")

    """
     don't need these in-memory approaches now, generator class is better...
     fwlyrlist = load_lyrics(LYRICDIR, prefix="firstwave", sep_artists=True)
     lyrlist = load_tag_docs("rap_KendrickLamar.lyr")

     LL_rap = LyricalLou(gen_prefx='rap', folder=LYRICDIR)
     LL_rap.create_bow()
     LL_1stwav = LyricalLou(gen_prefx='firstwave')
     LL_1stwav.create_bow()
     LL_rock = LyricalLou(gen_prefx='rock')
     LL_rock.create_bow()
     LL_country = LyricalLou(gen_prefx='country')
     LL_country.create_bow()
     LL_alt = LyricalLou(gen_prefx='alternative')
     LL_alt.create_bow()
     LL_punk = LyricalLou(gen_prefx='punk')
     LL_punk.create_bow()
     LL_metal = LyricalLou(gen_prefx='metal')
     LL_metal.create_bow()  

     LL_all = LyricalLou(gen_prefx='all')
     LL_all.create_bow()

     """
    mm_rap = TaggerTed(gen='rap', lydir=LYRICDIR)
    # rapmdl = do_skip_gram(iterlyr=LL_rap, dim=200, thrds=4, passes=6, grpsz=8, maxv=42000)
    fullmdl = do_skip_gram(iterlyr=mm_rap, dim=200, thrds=4, passes=6, grpsz=8, maxv=42000)

    if demo_word_model:
        """
        demo section applies word embedding tools and shows off capabilities of the models
        
        I plan to modify it to demonstrate one or two specific use cases rather than
        simply demonstrating random capabilities
        """
        from sklearn.decomposition import PCA
        print("\n---- vector model demonstration section ----")

        if fullmdl:
            fullmdl.predict_output_word(context_words_list=["homies", "friends",
                                                            "rollin", "dirty"], topn=5)
            X = fullmdl[fullmdl.wv.vocab]
            pca = PCA(n_components=2)
            result = pca.fit_transform(X)
            # convert from matplot bullshit: result[:, 0], result[:, 1]
            words = list(fullmdl.wv.vocab)
            pca_dct: dict = {}
            for i, word in enumerate(words):
                pca_dct[word] = xy=(result[i, 0], result[i, 1])

            result = fullmdl.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
            print("classic nlp analogy: man is to woman as king is to: %s" % result)

            more_examples = ["he his she", "big bigger bad"]
            for example in more_examples:
                a, b, x = example.split()
                predicted = fullmdl.most_similar([x, b], [a])[0][0]
                print("%s is to %s as %s is to %s" % (a, b, x, predicted))

            mismatched: list = ["breakfast", "cereal", "dinner", "lunch"]
            fullmdl.doesnt_match(mismatched)

            fullmdl.wv.most_similar("music")

            analogies: list = [
                {'positive': ['enemies', 'love'], 'negative': ['friends']},
                {'positive': ['friend', 'lover'], 'negative': ['fuck']},
                {'positive': ['woman', 'strong'], 'negative': ['man']}
            ]
            for example in analogies:
                if isinstance(example, dict):
                    pos: list = example.get('positive')
                    neg: list = example.get('negative')
                    print(fullmdl.wv.most_similar(positive=pos, negative=neg, topn=3))

        # k-means clustering to see how word and phrase vectors group

if phrases_train_test:

    common_wrds: dict = {}
    for wrd, ct in mm_rap.word_artists.items():
        if ct < 5:
            common_wrds[wrd] = 1

    print("   -- Creating and Testing phrases modeling in gensim --")
    from gensim.models.phrases import Phrases

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

    phrase_mdl = Phrases(train_phrase_corp, min_count=4, delimiter="-")
    country_phrases: dict = {}
    for phrase, score in phrase_mdl.find_phrases(test_phrase_corp).items():
        country_phrases[phrase] = score
        print(phrase, score)

if phrases_test2:
    train_phrase_rap: list = []
    test_phrase_rap: list = []
    test_tagsource_rap: list = []
    mm_rap.stream_td = False
    for i, td in zip(range(3200), mm_rap):
        if (i % 5) != 0:
            train_phrase_rap.append(td[0])
    for i, td in zip(range(3200), mm_rap):
        if (i % 5) == 0:
            test_phrase_rap.append(td[0])
            test_tagsource_rap.append(td[1])
    mm_rap.stream_td = True

    rap_phrase_mdl = Phrases(train_phrase_rap, min_count=4, delimiter="-")
    rap_phrases: dict = {}
    for phrase, score in rap_phrase_mdl.find_phrases(test_phrase_rap).items():
        rap_phrases[phrase] = score
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
    print("  ---- CREATE LDA model for topic modeling ----\n")
    # fw_sent = load_lyrics(folder=LYRICDIR, prefix='rap')

    dtmatrix, dterms = get_gdict_and_doc2bow(mm_rap)
    dterms.filter_extremes(no_below=1, no_above=0.5, keep_n=None)
    print("length of lda Dictionary after tfidf pruning is %d" % len(dterms))
    lda_mdl = run_lda_model(dtmatrix, dterms, topics=20, train_iter=100, word_topics=True)
    # lda_mdl.top_topics(texts=fw_sent)

    if lda_testcases:
        print("  ---- DISPLAY TEST CASES FOR LDA model ----\n")
        tot_topics = 5
        # most relavent words which make up a topic
        tot_words = 20
        print(lda_mdl.print_topics(num_topics=tot_topics, num_words=tot_words))
        # get topic distributions
        topic_dist = lda_mdl.state.get_lambda()

        # get topic terms
        for i, topic in lda_mdl.show_topics(formatted=True, num_topics=tot_topics, num_words=tot_words):
            print(str(i) + ": " + topic)
            print()

        # to show the top topic for a particular document by doc id:
        lda_mdl[dtmatrix[0]]

        testtok = [['he', 'brushed', 'em', 'walked', 'back', 'kentucky', 'fried', 'chicken',
                    'chicken', 'spot', 'was', 'light', 'skinned', 'nigga', 'talked', 'lot',
                    'curly', 'top', 'gap', 'teeth,', 'he', 'worked', 'window']]
        other_bow = [dterms.doc2bow(txt) for txt in testtok]
        test_doc = other_bow[0]
        # ldamdl_upd = lda_mdl.update(other_bow)

