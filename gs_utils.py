"""
gs_utils are common utlity functions for the app, such as file io.
"""
import json
import re
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from os.path import join
from os.path import exists, isfile, splitext
from os import listdir
from gs_datadict import OUTDIR, MODELDIR, LYRICDIR, ARTISTDIR, GENRES, CORESTOPS, GS_CONTRACT, \
    STOP_LYRICS, GS_STOP, CHAR_CONVERSION, QUOTEDASH_TABLE
import matplotlib.pyplot as plt
from wordcloud import WordCloud

trans_tmptbl = str.maketrans(CHAR_CONVERSION)

def clean_name(rawstr: str):
    """
    small wrangling utility to clean up artist names prior to searching on genius
    """
    cleanstr: str = rawstr.translate(trans_tmptbl)
    finalstr: str = cleanstr.translate(QUOTEDASH_TABLE).replace(" ", "")

    return finalstr

def get_files_list(ldir: str=LYRICDIR, incl_pre: list=None, excl_lst: list=None):
    """
    this is a simple file selector, creating a list that can be used by several tasks
    :param ldir: folder where lyrics files reside, my convention is LYRICDIR,
        usually STR (folder name), BUT can specify LIST of folders if necessary
    :param incl_pre: specify one or more genres for selecting artists, for example
        'rap' -> 'rap_KendrickLamar.lyr', 'rap_OutKast.lyr'..., or 'rock' or 'punk' etc.
    :param excl_lst: identify genre prefix(es) NOT to include, can specify prefix to
        exclude or specific artists to exclude
    :return: list of artist lyric files or None if error occurred
    """
    selected: list = []
    selectednum: int = 0
    if isinstance(ldir, str):
        # easier to deal with as a list
        dirs: list = [ldir]
    elif isinstance(ldir, list):
        dirs: list = ldir

    if isinstance(incl_pre, str):
        if incl_pre in GENRES:
            prefixes: list = [incl_pre]
        elif incl_pre == 'all':
            prefixes: list = GENRES
            print("get_files: including all genres")
        else:
            print("get_files: prefix %s not a recognized genre, Exiting" % incl_pre)
            return None
    elif isinstance(incl_pre, list):
        prefixes: list = []
        for pre in incl_pre:
            if pre in GENRES:
                prefixes.append(pre)
            elif pre == 'all':
                prefixes = GENRES
                print("get_files: including all genres")
            else:
                print("get_files: %s not a recognized genre, exiting" % pre)
                return None
    else:
        prefixes: list = GENRES

    if excl_lst:
        if isinstance(excl_lst, str):
            if excl_lst in prefixes:
                prefixes.remove(excl_lst)
                print("get_files: excluded %s genre" % excl_lst)
        else:
            for excl in excl_lst:
                if excl in prefixes:
                    prefixes.remove(excl)
                    print("get_files: excluded %s genre" % excl)

    numfil = len(prefixes)
    print("get_files: searching %d folders for %d genres --" % (len(dirs), numfil))

    for dirx in dirs:
        print("get_files: searching files in %s" % dirx)
        for fname in listdir(dirx):
            sep_loc = fname.find("_")
            dot_loc: int = fname.find(".")
            if sep_loc != -1:
                genre = fname[:sep_loc]
                artist: str = fname[sep_loc + 1: dot_loc]
            else:
                genre = "none"
                artist = "none"

            if genre not in prefixes:
                print("  %s genre not included, artist: %s, file: %s" % (genre, artist, fname))
                continue
            if excl_lst:
                if artist in excl_lst:
                    print("  %s - excluded artist, file: %s" % (artist, fname))
                    continue
            fqfil = join(dirx, fname)
            if isfile(fqfil):
                file_splits = splitext(fqfil)
                try:
                    file_splits.index('.lyr')
                except ValueError:
                    print("  get_files: not a lyrics file, file: %s" % fname)
                    continue
                else:
                    selected.append({"file": True, "name": fname, "path": fqfil})
                    selectednum += 1
                    # print("get_files: added %s to filter" % artist)
            else:
                print("  %s is not a file, continuing with folder" % fqfil)

    print("  -- get_files: COMPLETED, %d files selected --\n" % selectednum)

    return selected

def filter_for_end_of_track(lyf: str, ldir: str=LYRICDIR, eot: str="\nSONGBREAK\n"):
    """
    standardizes end-of-track indicator for lyrics files, can be used for other filters
    but this is a DESIGN DECISION with DOWNSTREAM IMPLICATIONS.
      Alter corpus source files ONLY if ALL downstream processes, known or planned, have
      the need for it! (i.e. you can't unring this bell)
        I scrub text 'just-in-time' (JIT), in-line with generator that streams lyrics
        Exception- this Fx adds a delimiter between songs and removes blank lines
    :param lyf: str filename for lyrics file
    :param ldir: str folder where lyf resides, expected to be with other lyrics
    :param eot: str to embed in .lyr file as marker for end of track
    :return: 0 if success, 13 if fail or known error
    """

    fq_lyf = join(ldir, lyf)
    if exists(fq_lyf):
        f_h = open(fq_lyf, mode="r")
        contents: str = f_h.read()
        f_h.close()
        sizein: int = len(contents)
        seppos: int = lyf.find("_")
        extpos: int = lyf.find(".")
        artst: str = lyf[seppos + 1: extpos]
        mark: str = "EmbedShare URLCopyEmbedCopy"
        mtch = re.split(mark, string=contents)

        if len(mtch) < 2:
            mark = "songbreak"
            mtch = re.split(mark, string=contents)
            if len(mtch) > 1:
                contents = re.sub(mark, repl=eot, string=contents)
                print("converted %d songbreak to eot in %s lyrics" % (len(mtch), artst))
            else:
                mark = re.sub(r"\n", repl="", string=eot)
                mtch = re.split(mark, string=contents)
                if len(mtch) > 1:
                    print("%s has %d %s delimiters, u good bro, skipping this one"
                          % (artst, len(mtch), mark))
                else:
                    mark = r"(?m)^\n\n"
                    mtch = re.split(mark, string=contents)
                    if len(mtch) > 1:
                        contents = re.sub(mark, repl=eot, string=contents)
                        print("converted %d double-blank-line to eot in %s lyrics" % (len(mtch), artst))
                    else:
                        print("\nError: no known song breaks in %s lyrics, EDIT MANUALLY" % artst)
                        return 13
        else:
            contents = re.sub(mark, repl=eot, string=contents)
            eotx = re.sub(r"\n", repl="", string=eot)
            print("converted %d %s to %s in %s lyrics" % (len(mtch), mark, eotx, artst))

        # set re MULTILINE mode to match caret (^) at start of each 'line'
        mtch = re.split(r"(?m)^\n", string=contents)
        if len(mtch) > 1:
            contents = re.sub(r"(?m)^\n", repl="", string=contents)
            print("filter_lyrics: removed %d blank lines in %s lyrics" % (len(mtch), artst))
        else:
            print("no blank lines for %s" % artst)

        f_h = open(fq_lyf, mode="w", newline="")
        sizeout = f_h.write(contents)
        f_h.close()
        print("filter_lyrics for %s - %d chars read in, %d written out" % (artst, sizein, sizeout))
        return 0

    else:
        print("filter_lyrics ERROR: could not locate %s file" % fq_lyf)
        return 13

def clean_lyrics_file(lyf: str):
    """
    read lyrics corpus (*.lyr file), scrub it, and write it back to a new file (append a
    version number to end of file).  perform the following edits on each line of file:
        1. remove punctuation and convert text to lower case
        2. removes words that match CORESTOPS (my appended list based on gensim eng stoplist)
        3. expands contractions, as per my contraction dictionary GS_CONTRACT
    :param lyf: str name of lyrics corpus
    :return: list of str for each track in lyrics corpus
    """
    ext_pos = lyf.find(".")
    lyfpre = lyf[:ext_pos]
    lyfpost = lyf[ext_pos:]
    fq_file = join(OUTDIR, lyf)
    fq2 = OUTDIR + lyfpre + "_cln" + lyfpost
    re_punc = re.compile(r'[.?!;:<>,(){}]')
    if exists(fq_file):
        f2_h = open(fq2, mode='w', newline="")
        for line in open(fq_file, mode='r', newline=""):
            if len(line) < 5:
                continue
            else:
                tmp = line.lower()
                for wrd, expand in GS_CONTRACT.items():
                    tmp = re.sub(wrd, repl=expand, string=tmp)
                for wrd, cnvrt in CHAR_CONVERSION.items():
                    tmp = re.sub(wrd, repl=cnvrt, string=tmp)
                line = re_punc.sub(repl="", string=line).split()
                line = " ".join(w for w in line if w not in CORESTOPS)
                f2_h.write(line)
        f2_h.close()
    print("\ncompleted cleaning corpus file: %s" % lyfpre)

    return

def read_tokens_to_list(lyf: str, wrdtokens: bool=True):
    """
    reads lines from a lyrics corpus file into python list- splits words to separate tokens
    -- memory-intensive! better alternative is to use an iterator-generator object
    -- this simple utility also assumes that the lyrics have already been scrubbed
    :param lyf: str name of lyrics file, assumed to be located in app OUTDIR
    :param wrdtokens: if True, splits each line into list of words
    :return: list of lyrics lines
    """
    lyriclst: list = []
    fqf = join(OUTDIR, lyf)
    f_h = open(fqf, mode="r", newline="")
    for line in f_h:
        if len(line) > 5:
            if wrdtokens:
                lyriclst.append(line.split())
            else:
                lyriclst.append(line)
    f_h.close()

    return lyriclst

def extract_words(sentstr, stops):
    """
    simple utility to word tokenize a text string and remove stop words
    :param sentstr: input string of text
    :param stops: list of stop words to remove
    :return:
    """
    words = re.sub(r"[^\w]", " ", sentstr).split()
    cleaned_text = [w.lower() for w in words if w.lower() not in stops]
    new_s: str = ""
    for x in cleaned_text:
        new_s = new_s + " " + x
    return new_s

def do_stops(twlst: list, stop1: list=GS_STOP, stop2: list=STOP_LYRICS):
    """
    do_stops is preprocessing function to remove word tokens based on a stop list
    :param twlst: list of list, list of dict, or list of str for Tweets
    :param stop1: list of stop words, defaults to GS_STOP
    :param stop2: list of stop words, defaults to STOP_ADD
    :return: list of tweets with word tokens and stop words removed
    """
    clean_list: list = []
    for twis in twlst:
        if isinstance(twis, list):
            tmp_wrds: list = [cw for cw in twis if cw not in stop1]
            if stop2 is not None:
                clean_list.append([cw for cw in tmp_wrds if cw not in stop2])
            else:
                clean_list.append(tmp_wrds)
        else:
            if isinstance(twis, dict):
                twemp: list = str(twis['text']).split()
            else:  # assume isinstance(twis, str)
                twemp: list = twis.split()

            tmp_wrds: list = [cw for cw in twemp if cw not in stop1]
            if stop2 is not None:
                clean_list.append(' '.join([str(cw) for cw in tmp_wrds if cw not in stop2]))
            else:
                clean_list.append(' '.join([str(cw) for cw in tmp_wrds]))

    return clean_list

def get_stop_file(stf: str):
    """
    this reads stopfile from the nlp models folder and returns it as a set
    :param stf: file name, to be prepended with MODELDIR
    :return: set of stop words
    """
    tmpset: set = set()
    fqfil = MODELDIR + stf
    f_h = open(fqfil, mode="r")
    for line in f_h:
        tmpset.add(line)

    adhoc: list = ['ha', 'oh', 'get', 'got', 'one', 'say', 'like', 'la', 'go', 'say',
                   'said', 'yeah', 'ya', 'u', 'da', 'give', 'thing', 'way', 'really',
                   'let', 'whoa', 'hey', 'na', 'put', 'doo', 'doo', 'let', 'fa', 'cata',
                   'hi', 'so', 'little', 'dit', 'songbreak', 'could']
    for wrd in adhoc:
        tmpset.add(wrd)

    return tmpset

def append_genre_registry(albdct: list, genre: str, outd: str=OUTDIR):
    """
    save a unique track ID (artist name + 3-digit counter) plus name of track for each song
    in the lyrics corpus, one artist file for each genre.
    entries are appended when this is called, so there is logic to verify that the artist's
    tracks have not already been added.
    PURPOSE: this allows TaggedDocument tags used in topic modeling to be mapped back to
        an artist and song- for example, if providing test lyrics and looking for the most
        similar lyrics, it is desirable to be able to return similar artists-songs!
    :param albdct: all artist's albums as list of dict
        'albumid': int, 'title': str, and for each track- song ID: song URL
    :param genre: name of genre registry to create or append
    :param outd: output directory, defaults to app OUTDIR
    :return 0 if save OK
    """

    if not genre in GENRES:
        print("save_artist_file: use out file= '<genre>_registry.json'! exiting")
        return 1
    else:
        outf: str = genre + "_registry.json"

    fqf = join(outd, outf)
    with open(fqf, mode='a') as f_h:
        if isinstance(albdct, list):
            for alb in albdct:
                json.dump(alb, f_h)
                f_h.write("\n")
        elif isinstance(albdct, dict):
            # typical course when processing after get_lyrics is run in build_corpus
            json.dump(albdct, f_h)
            f_h.write("\n")
            for ad, i in zip(albdct, range(1)):
                print("append_genre_registry: added tracks for %s" % ad[:-3])

    return 0

def retrieve_genre_registry(genre: str, indir: str=ARTISTDIR):
    """
    retrieve track lists for artists, useful in matching up document ID's when doing
    topic modeling and other doc2vec tasks with song lyrics
    :param genre: each genre has a registry file made up of track dicts for each artist
    :param indir: folder where json registry files reside, default is ARTISTDIR
    :return:
    """
    infile = genre + "_registry.json"
    fqf = join(indir, infile)
    if isfile(fqf):
        reg_lst: list = []
        with open(fqf, mode="r") as f_h:
            for line in f_h:
                reg_lst.append(json.loads(line))

        print("  retrieved %s registry with %d artist track lists" % (genre, len(reg_lst)))
        return reg_lst
    else:
        print("retrieve_genre_registry ERROR: %s not valid!" % fqf)
        return None

def get_artists_in_genre(incl_genres, lydir: str=LYRICDIR):
    """
    creates and displays a list of all artists for genres selected via incl_genres parm.
    selects all genres if incl_genres is left blank or if 'all' is passed as str or list
    :param incl_genres:
    :param lydir: folder for lyric files
    :return:
    """
    print("  -- create_genre_lists finds all artists with lyrics in genre --")
    if not incl_genres:
        search_gen = GENRES
    else:
        if incl_genres == "all":
            search_gen = GENRES
        else:
            if isinstance(incl_genres, str):
                if incl_genres in GENRES:
                    search_gen = [incl_genres]
            elif isinstance(incl_genres, list):
                search_gen: list = []
                for genx in incl_genres:
                    if genx in GENRES:
                        search_gen.append(genx)

    total_gen: int = len(search_gen)
    art_added: int = 0
    genre_artists: dict = {}
    genre_artists = genre_artists.fromkeys(search_gen)

    def add_artist(gn: str, ar: str, g_a: dict):
        """
        inner function to insert artist in appropriate dict for genre
        :param gn: str with genre
        :param ar: str with artist
        :param g_a: dict of genre-artists
        :return: 0 if artist inserted, 1 if not
        """
        found: bool = False
        for k, v in g_a.items():
            if k == gn:
                found = True
                if not v:
                    artsts: set = {ar}
                elif isinstance(v, list):
                    artsts: set = set(v)
                elif isinstance(v, set):
                    artsts: set = v

                if not ar in artsts:
                    artsts.add(ar)
        if found:
            g_a[gn] = artsts
        return g_a, found

    for fname in listdir(lydir):
        fqfil = join(lydir, fname)
        if isfile(fqfil):
            sep_pos = fname.find("_")
            ext_pos = fname.find(".")
            gen_pre = fname[:sep_pos]
            art_nam = fname[sep_pos + 1: ext_pos]
            if gen_pre in search_gen:
                genre_artists, was_added = add_artist(gen_pre, art_nam, genre_artists)
                if was_added:
                    art_added += 1

    print("      create_genre added %d artists across %d genres" % (art_added, total_gen))
    return genre_artists

def save_artist_info(artdct: dict, genre: str, outd: str = OUTDIR):
    """
    this is an optional method to save the artist info dict that is simply created as a
    step towards pulling albums and lyric tracks, it may allow recreation of a corpus
    from scratch easier.
    :param artdct: artist name, genius id, img file, lyric file, etc.
    :param genre: name of genre for this artist
    :param outd: output directory, defaults to app OUTDIR
    :return 0 if save OK
    """
    outf = genre + "_artistdb.info"
    fqf = join(outd, outf)
    with open(fqf, mode='a') as f_h:
        if isinstance(artdct, dict):
            json.dump(artdct, f_h)
            f_h.write("\n")

    return 0

def scrub_cloud(twl: list):
    """
    scrub_text can perform numerous text removal or modification tasks on tweets,
    there is tweet-specific content handled here which can be optionally commented out
    if resulting corpus loses too much detail for downstream tasks like sentiment analysis

    :param twl: list of tweets
    :return: list of words OR str of words if rtn_list= False
    """
    from gs_datadict import GS_STOP, STOP_LYRICS, CHAR_CONVERSION
    import re

    for tweetxt in twl:
        if isinstance(tweetxt, str):
            # remove newlines in tweets, they cause a mess with many tasks
            tweetxt = tweetxt.replace("\n", " ")
            # remove standalone period, no need in a tweet
            tweetxt = re.sub(r"\.", " ", tweetxt)
            # expand contractions using custom dict of contractions
            tweetxt = re.sub(GS_STOP, "", tweetxt)
            # convert ucs-2 chars appearing in english tweets
            tweetxt = re.sub(STOP_LYRICS, "", tweetxt)
            # parsing tweet punc: don't want to lose sentiment or emotion
            tweetxt = re.sub(CHAR_CONVERSION, "", tweetxt)
            # parameter "utf-8" escapes ucs-2 (\uxxxx), "ascii" removes multi-byte
            binstr = tweetxt.encode("ascii", "ignore")
            tweetxt = binstr.decode()
            # now, tokenize and run words through cleaning:
            splitstr = tweetxt.split()
            stokens: list = []
            for w in splitstr:
                if len(w) < 2:
                    continue
                # lower case all alpha words
                if str(w).isalpha():
                    w: str = str(w).lower()
                    stokens.append(w)
            splitstr: list = []
            for wrd in stokens:
                wrd = wrd.strip()
                splitstr.append(wrd)

    return splitstr

def get_cloud_input(batch_words):
    """
    get lyrics token lists from object and process to prep for cloud
    :param batch_words:
    :return:
    """
    wrdct: dict = {}
    cur_artist: str = ""
    for wrds, artst in batch_words:
        for word in wrds:
            if wrdct.get(word):
                wrdct[word] += 1
            else:
                wrdct[word] = 1
        if not cur_artist.startswith(artst):
            if not cur_artist == "":
                print("streaming done for %s" % artst)
            cur_artist = artst

    return wrdct

def do_cloud(cloud_words: list, opt_stops: list = None, maxwrd: int = 120):
    """
    wordcloud package options can be explored via '?wordcloud' (python- show docstring)
    background_color="white" - lighter background makes smaller words more legible,
    max_words= this can prevent over clutter, mask=shape the cloud to an image,
    stopwords=ad-hoc removal of unwanted words, contour_width=3,
    :param cloud_words: list of list of word tokens
    :param opt_stops: list of words (str) to remove prior to creating wordcloud
    :param maxwrd: int typically from 80 to 120 for total words to appear in cloud
    :return:
    """
    from wordcloud import WordCloud
    import io
    import matplotlib.pyplot as plt

    cloud_text = io.StringIO(newline="")
    for tok in cloud_words:
        if isinstance(tok, str):
            cloud_text.write(tok + " ")
        else:
            for a_tw in tok:
                if isinstance(a_tw, list):
                    cloud_text.write(" ".join([str(x) for x in a_tw]) + " ")
                if isinstance(a_tw, str):
                    # if simple list of text for each tweet:
                    cloud_text.write(a_tw + " ")

    wordcld = WordCloud(width=800, height=800, max_words=maxwrd,
                        background_color='white',
                        stopwords=opt_stops, min_word_length=4,
                        min_font_size=10).generate(cloud_text.getvalue())

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcld)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    return

def check_outliers(lobj):
    """
    goes through a lyrics object and checks for few extreme values for
    calculated attributes such as tf and idf
    """
    artvals: list = []
    artistlst: list = list(lobj.artists.copy())
    artistct: int = len(artistlst)
    for i in range(artistct):
        if lobj.tfidf_artist.get(artistlst[i]):
            artvals.append(list(lobj.tfidf_artist[artistlst[i]].values()))

    for i in range(artistct):
        hicut = np.percentile(a=artvals[i], q=98)
        locut = np.percentile(a=artvals[i], q=2)
        voc_size = len(artvals[i])
        removed: int = 0
        for_deletion: list = []
        tmpdct: dict = lobj.tfidf_artist[artistlst[i]]
        for wrd, tfival in tmpdct.items():
            if tfival > hicut or tfival < locut:
                for_deletion.append(wrd)
        if for_deletion:
            for wrd in for_deletion:
                tmpdct.pop(wrd)
                removed += 1

        lobj.tfidf_artist[artistlst[i]] = tmpdct
        pct_del = round((removed / voc_size), ndigits=3)
        print("outliers removed %d tfidf_artist, %.3f percent, for %s" % (removed, pct_del, artistlst[i]))

    return artistlst, artvals

def plot_artist_tfidf(lobj, artst: str):

    artvals: list = list(lobj.tfidf_artist[artst].values())
    fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    finalmin: float = 0.0006
    tfimed = np.median(artvals)
    tfistd = np.std(artvals)
    finalmax = tfimed + (3 * tfistd)
    xrang = finalmax - finalmin
    num_bins = int(round(xrang / 0.0002, ndigits=0) + 2)

    ax1.set_xlim(finalmin, finalmax)
    plt.xlim(finalmin, finalmax)

    ax1.hist(artvals, bins=num_bins, density=False, range=(finalmin, finalmax))

    plt.ylabel('Occurences')
    plt.xlabel('tfidf value')
    plt.suptitle("tfidf for words of %s" % artst)
    plt.show()

    return

def plot_artist_tf(lobj, artst: str):
    artvals: list = list(lobj.tf_by_artist[artst].values())
    tot_wrds: int = len(artvals)

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    finalmin: float = 0.002
    tfimed = np.median(artvals)
    tfistd = np.std(artvals)
    finalmax = tfimed + (3 * tfistd)
    xrang = finalmax - finalmin
    num_bins = int(round(xrang / 0.0002, ndigits=0) + 2)

    ax1.set_xlim(finalmin, finalmax)
    plt.xlim(finalmin, finalmax)
    ax1.hist(artvals, bins=num_bins, density=False, range=(finalmin, finalmax))
    plt.ylabel('Occurences')
    plt.xlabel('tfidf value')
    plt.suptitle("tf for %d words for artist: %s" % (tot_wrds, artst))
    plt.show()

    return

def compare_artist_tfidf(lobj, artistlst: list = None):
    """
    within a lyrics object, compare high-tf words between artists
    ASSUMES TWO ARTISTS WILL ALWAYS BE NAMED!
    """
    num_bins: int = 200
    artvals: list = []
    plots: int = 0
    for artist in artistlst:
        artvals.append(list(lobj.tfidf_artist[artist].values()))
        plots += 1

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=plots)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    totmx: float = 0.09
    totmn: float = 0.002
    for i in range(plots):
        tfimx = np.max(artvals[i])
        tfimn = np.min(artvals[i])
        if i == 0:
            ax1.set_xlim(tfimn, tfimx)
            ax1.hist(artvals[i], num_bins, density=False, range=(tfimn, tfimx))
        else:
            ax2.set_xlim(tfimn, tfimx)
            ax1.hist(artvals[i], num_bins, density=False, range=(tfimn, tfimx))

    # color1 = np.random.uniform(15, 80, len(art1_vals))
    # the histogram of the data
    # n, bins, patches = ax1.hist(artvals[0], num_bins, density=False, range=(totmn, totmx))
    # n2, bins2, patches2 = ax2.hist(artvals[1], num_bins, density=False, range=(totmn, totmx))

    # add a 'best fit' line
    # mu = np.mean(artvals[0])  # mean of distribution
    # sigma = np.std(artvals[0])  # standard deviation of distribution
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #     np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    # ax1.plot(bins, y, '--')

    # fig, ax = plt.subplots()
    # ax.hist(art1_vals, bins=30, linewidth=0.01, edgecolor="white")
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.ylabel('Occurences')
    plt.xlabel('tfidf value')
    plt.suptitle("tfidf for words of %s vs %s" % (artistlst[0], artistlst[1]))
    plt.show()

    return
