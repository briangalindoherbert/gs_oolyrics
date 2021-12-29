"""
file gs_genius_wrappers.py: contains functions to pull lyrics for songs to create corpora for NLP
get a token by signing up at genius.com, use token to access the api.

encountering issues with a timeout when making requests to api.genius.com
may need to use something like no_timeout = Timeout(connect=None, read=None)
TODO: learn and apply settings to increase connection timeout

Genius API overview:
- sign up to get an access token, used as Oauth2.0 Bearer_Token authorization method
    then: lg_this = lyricsgenius.Genius(gen_token)
- to get the ID for an artist, search for artist by name, as in:
https://api.genius.com/search?q=Grateful%20Dead
- this returns a numeric ID for artist, in this case: 21900
- then use artist_albums api call to get lists of albums, and pull lyrics per album
- can also search for song lyrics by genre
     the genre id for rock is: 567
- example hierarchy of calls:
    lg_this = lyricsgenius.Genius(gen_token)
    albums1 = lg_this.artist_albums(21900, per_page=50, page=1)
    or the_Dead = lg_this.artist(21900)
    then albums2 = the_Dead.get_album_list()
"""

import socket
import io
from os import listdir
from os.path import join
from simplejson.errors import JSONDecodeError
from urllib3.connection import HTTPConnection
from urllib3.exceptions import ReadTimeoutError
from requests.exceptions import Timeout, HTTPError
from lyricsgenius import Genius
from gs_datadict import OUTDIR, LYRICDIR, CHAR_CONVERSION
from gsGenSim import line_cleanup

char_strip_table = str.maketrans({" ": None, "/": "-", ".": "", "'": "", "–":"-"})
unicode_translate = str.maketrans(CHAR_CONVERSION)

def get_artist_info(l_g, namestr: str = None, artid: int = None):
    """
    establish general information about artist, either from a text search for name or
    a provided genius artist ID.  returns info like name, id, url, number of followers,
    number of annotations,
    :param l_g: instantiated Genius object
    :param namestr: name of band or artist
    :param artid: if known, provide genius ID for this artist or band
    :return: artist dict with general information
    """

    def get_artist_by_name(artnam: str):
        """
        inner function to allow multiple tries as name search for artist can fail due to
        capitalization inconsistencies, use or non-use of prepositions, and even the
        occasional null-sized space unicode character (\u200B)
        :param artnam:
        :return:
        """
        raw = l_g.search_artists(search_term=artnam, per_page=10)
        hits = raw['sections'][0]['hits']
        rawdict: dict = {}
        good_id: int = 0
        for x in hits:
            if 'name' in x['result']:
                if x['result']['name'] == artnam:
                    good_id = x['result']['id']
                    rawdict = l_g.artist(artist_id=good_id)
                    return rawdict['artist'], good_id
        # best if name matches, but if not, first found is prob the one
        for x in hits:
            if 'name' in x['result']:
                good_id = x['result']['id']
                rawdict = l_g.artist(artist_id=good_id)
                return rawdict['artist'], good_id

        return None

    if artid:
        artdict = l_g.artist(artist_id=artid)
        good_name: str = artdict['artist']['name']
        found_id = artid
        artdict = artdict['artist']
    elif namestr:
        artdict, found_id = get_artist_by_name(artnam=namestr)
        if not artdict:
            # try again with zero-spaced space
            wack_name = "\u200B" + namestr
            artdict, found_id = get_artist_by_name(artnam=wack_name)
            if not artdict:
                # try one more time, see if title case works
                wack_name = namestr.title()
                artdict, found_id = get_artist_by_name(artnam=wack_name)
                if not artdict:
                    print("couldn't find artist match for %s, TRY DIFFERENT SPELLING" % namestr)
                    return None
        good_name: str = str(artdict['name']).translate(unicode_translate)

    if artdict:
        artd: dict = {"id": found_id, "name": good_name, "url": artdict['url'],
                      "following": artdict['followers_count'], "verified": artdict['is_verified'],
                      }
    if artdict['image_url']:
        artd['img'] = artdict['image_url']

    if artdict['user']:
        if 'role_for_display' in artdict['user']:
            artd['artist_role'] = artdict['user']['role_for_display']

    return artd

def get_artist_albums(l_g, artst, albmax: int=24):
    """
    for a given artist, return list of albums with genius album ID, plus songID's and songURL's
    for each track on each album.
    if only an artist id is passed in, Fx tries to pull artist name from album info returned
    album dict returned from this Fx is iterated through by get_lyrics to create corpus for artist
    :param l_g: instantiated Genius object
    :param artst: this can be a Genius ID int, or the artist dict from get_artist_info
    :param albmax: maximum albums to return
    :return: dict: 'albumid', 'title', plus [song_id: song_url]
    """
    if isinstance(artst, dict):
        # best case, should have artist ID and name
        art_id: int = artst['id']
        art_name: str = artst['name']
    elif isinstance(artst, int):
        art_id: int = artst
        art_name: str = ""
    else:
        print("get_artist_album: invalid value for artst, see docstring")
        return None

    lg_return: dict = l_g.artist_albums(artist_id=art_id, per_page=albmax)
    album_lst: list = lg_return['albums']
    if not art_name:
        if 'artist' in album_lst[0]:
            art_name = album_lst[0]['artist']['name']
        elif 'artist' in album_lst[1]:
            art_name = album_lst[1]['artist']['name']
        elif 'artist' in album_lst[2]:
            art_name = album_lst[2]['artist']['name']
        else:
            art_name = "unk"

    albtrx: list = []
    trx_cnt: int = 0
    for alb in album_lst:
        if isinstance(alb, dict):
            tmpdct: dict = {"artist": art_name, "artid": art_id, "albumid": alb['id'], "title": alb['name']}
            x = l_g.album_tracks(album_id=alb['id'], per_page=28)
        x = x['tracks']
        trx_cnt += len(x)
        for sng in x:
            tmpdct[sng['song']['id']] = sng['song']['url']
        albtrx.append(tmpdct)

    print("get_artist_albums: %d albums and %d trakID's for %s" % (len(albtrx), trx_cnt, art_name))

    return albtrx

def get_albumtrax(l_g, a_d: dict, band: str=""):
    """
    get_albumtrax iterates through a dictionary of album id's and pulls the track detail
    for each album.
    :param l_g: instantiated lyricgenius Genius object, created with bearer token
    :param a_d: dictionary of key: album_id and value: album title
    :param band: str name of artist or band, used for print display
    return: songs (with song IDs) for each album
    """
    trax_d = {}
    for album in a_d.keys():
        # get list of songs and then lyrics
        try:
            x = l_g.album_tracks(album)
        except ConnectionResetError:
            try:
                x = l_g.album_tracks(album)
            except ConnectionResetError:
                x = l_g.album_tracks(album)
        finally:
            if x:
                trax_d[album] = x
    print("get_albumtrax- searched %d albums by %s" %(len(trax_d), band))
    print("              and found %d songs \n" % sum(trax_d.values()))
    return trax_d

def get_lyrics_from_trax(l_g, trax: dict):
    """
    get_lyrics_from_trax uses what's returned from get_albumtrax to pull lyrics for album
    early effort- I've replaced this with 'get_lyrics' and an improved workflow
    :param l_g: lyric genius object instance
    :param trax: dictionary accessed as track_dict[album id]['tracks']
    :return str of lyrics for all songs in album
    """
    lyricio = io.StringIO(newline='')
    # iterate through albums then tracklists, get lyrics for each song id
    this_tracks = trax['tracks']
    for this_song in iter(this_tracks):
        song_dict = this_song['song']
        lyricio.write(song_dict['title'] + "\n")
        try:
            lyricio.write(l_g.lyrics(song_dict['id']))
        except ConnectionResetError:
            try:
                lyricio.write(l_g.lyrics(song_dict['id']))
            except ConnectionResetError:
                lyricio.write(l_g.lyrics(song_dict['id']))
        finally:
            lyricio.write("\n\n")

    album_lyrics: str = lyricio.getvalue()
    lyricio.close()

    return album_lyrics

def save_lyrics(the_words: str, the_file: str):
    """
    write lyrics to a file
    Args:
        the_words: str containing lyrics
        the_file: str name of file to save
    Returns: 0 if successful
    """
    with open(the_file, mode='wt+', newline=None) as write_lyrics:
        write_lyrics.write(the_words)
        write_lyrics.close()
    return 0

def load_art_lyrics(prefix: str, artst: str="", lydir: str=LYRICDIR, artreg: list=None):
    """
    loads lyrics into in-memory list, specify genre prefix, artist name, artist's track registry
    :param prefix: str or list for musical genre(s) and .lyr file prefix to select
    :param artst: str artist name
    :param lydir: str folder name
    :param artreg: list of dict of track registries for artists
    :return list of str corpus
    """
    agg_wrds: int = 0
    traklst: list = []
    eottag: bool = False
    trak_ct: int = 0

    for fname in listdir(lydir):
        sep_loc = fname.find("_")
        fnamepre = fname[:sep_loc]
        if isinstance(prefix, str):
            if fnamepre == prefix:
                do_this: bool = True
            else:
                do_this: bool = False
        elif isinstance(prefix, list):
            if fnamepre in prefix:
                do_this: bool = True
            else:
                do_this: bool = False
        if do_this:
            dot_loc: int = fname.find(".")
            art_nam: str = fname[sep_loc + 1: dot_loc]
            art_nam = art_nam.replace(" ", "")
            if artst:
                if isinstance(artst, str):
                    if art_nam.startswith(artst):
                        fqf = join(lydir, fname)
                        f_h = open(fqf, mode="r")
                        trktag: str = art_nam + str(trak_ct).rjust(3, "0")
                        for line in f_h:
                            if len(line) > 5:
                                if line.startswith("SONGBREAK"):
                                    eottag: bool = True
                                    splts, agg_wrds = line_cleanup(txt=artreg[trktag], word_ct=agg_wrds)
                                    traklst.extend(splts)
                                else:
                                    splts, agg_wrds = line_cleanup(line, agg_wrds)
                                    traklst.extend(splts)
                            if eottag:
                                trktag: str = art_nam + str(trak_ct).rjust(3, "0")
                                yield traklst, trktag
                                traklst = []
                                eottag = False
                                trak_ct += 1
                        print("    end of lyrics for %s" % art_nam)
                        f_h.close()
    return

def get_cleantracks(t_d: dict):
    """
    return concise dict with song ID's and url from results returned from get_albumtrax()
    can use with a corpus of albums and tracks, this is pre-processing step before
    getting the lyrics for the songs
    :param t_d: the raw info on album tracks returned from get_albumtrax
    Returns: a_d
    """
    a_d = {}
    for albid, albdet in t_d.items():
        trk_dict: dict = {}
        for sng in albdet['tracks']:
            song_id = sng['song']['id']
            trk_dict[song_id] = sng['song']['url']
        a_d[albid] = trk_dict

    return a_d

def get_lyrics(song_d: list, l_g: Genius = None, genre: str='rock', lydir: str=OUTDIR):
    """
    get_lyrics returns a string of lyrics for list of albums for an artist
    song_d is now a list of dict: each dict has 'albumid', 'title' and then
    sequence of {song_id: song_url}
    :param song_d: list created by get_artist_albums: song ID's for each album
    :param l_g: instantiated Genius object
    :param genre: str with name for genre of this artist
    :return str of lyrics
    """
    HTTPConnection.default_socket_options = HTTPConnection.default_socket_options + [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 2), (socket.SOL_TCP, socket.TCP_KEEPINTVL, 12)
    ]
    agg_bytes: int = 0
    trax_reg: dict = {}
    trax_ctr: int = 0

    # strip last 7 chars for 'lyrics' and first 19 for 'https://...' to slice artist-track name
    end_pos = -7
    start_pos = 19
    this_artist: str = str(song_d[0]['artist']).translate(unicode_translate)
    this_artist = this_artist.translate(char_strip_table)
    outf: str = genre + "_" + this_artist + ".lyr"
    lyr_f = join(OUTDIR + outf)
    f_h = open(lyr_f, mode='w', encoding='utf-8', newline='')

    for alb in song_d:
        for sid, surl in alb.items():
            if isinstance(sid, str):
                if sid == 'artist':
                    continue
            elif isinstance(sid, int):
                # album dict: the only integer keys are track IDs, value is lyrics url
                if surl.endswith("lyrics"):
                    if isinstance(surl, str):
                        trak_nam = surl[start_pos: end_pos]

                    try:
                        tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                    except (HTTPError, JSONDecodeError):
                        print("caught 1st level exception: HTTP, retrying...")
                        try:
                            tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                        except (HTTPError, JSONDecodeError):
                            print("  caught 2nd exception: HTTP, retrying...")
                            try:
                                tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                            except (HTTPError, JSONDecodeError):
                                    print("    3rd successvie HTTP exception, bailing on %s" % surl)
                                    tmpstr: str = ""
                                    continue
                        except (Timeout, ReadTimeoutError):
                            print("  2nd level exception: Timeout-ReadTimeout, one last try")
                            try:
                                tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                            except:
                                print("    3rd level exception, bailing on %s" % surl)
                                tmpstr: str = ""
                                continue
                    except (Timeout, ReadTimeoutError):
                        # time.sleep(0.5)  add some delay into the read cycle
                        print("caught 1st level exceptions.Timeout, retrying")
                        try:
                            tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                        except (HTTPError, JSONDecodeError):
                            print("  caught 2nd-level exception- HTTP, retrying")
                            try:
                                tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                            except (HTTPError, JSONDecodeError):
                                print("    caught 3rd-level exception- HTTP, one more try")
                                try:
                                    tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                                except (HTTPError, JSONDecodeError):
                                    print("      4th level exception- HTTP, bailing on %s" % surl)
                                    tmpstr: str = ""
                                    continue
                            except (Timeout, ReadTimeoutError):
                                print("3rd level exception: Timeout- una vez más...")
                                try:
                                    tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                                except:
                                    print("4th level exception, bailing on %s" % surl)
                                    tmpstr: str = ""
                                    continue

                        except (Timeout, ReadTimeoutError):
                            print("caught 2nd successive Timeout, retrying")
                            try:
                                tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                            except (Timeout, ReadTimeoutError):
                                print("3rd successive Timeout exception- retrying")
                                try:
                                    tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                                except (Timeout, ReadTimeoutError):
                                    print("4th successive timeout, una vez más")
                                    try:
                                        tmpstr: str = l_g.lyrics(song_url=surl, remove_section_headers=True)
                                    except:
                                        print("5th timeout, bailing on %s" % surl)
                                        tmpstr: str = ""
                                        continue
                    finally:
                        # finally clause guaranteed to run even with an except clause continue
                        if tmpstr:
                            if len(tmpstr) > 80:
                                trak_siz = f_h.write(tmpstr)
                                agg_bytes += trak_siz
                                f_h.write("\n")
                                trak_key = this_artist + str(trax_ctr).rjust(3, "0")
                                trax_reg[trak_key] = trak_nam
                                trax_ctr += 1
                                print("get_lyrics characters written: %6d" % agg_bytes, sep='', end='\r', flush=True)

    f_h.close()
    print("wrote %7d characters for %s lyrics" % (agg_bytes, lyr_f))

    return trax_reg
