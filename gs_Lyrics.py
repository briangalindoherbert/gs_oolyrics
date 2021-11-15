"""
file gs_Lyrics.py: contains functions to pull lyrics for songs to create corpora for NLP
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

import lyricsgenius
import requests
import chardet

def get_albumtrax(l_g, a_d: dict):
    """
    get_albumtrax iterates through the dictionary of album id's and pulls the track detail
    for each album.
    :param l_g: lyric genius key
    :param a_d: artist dictionary
    return: songs on album
    """
    trax_d = {}
    for album in iter(a_d.keys()):
        # get list of songs and then lyrics
        try:
            x = l_g.album_tracks(album)
        except ConnectionResetError as E:
            try:
                x = l_g.album_tracks(album)
            except ConnectionResetError as E:
                x = l_g.album_tracks(album)
        finally:
            trax_d[album] = x
    return trax_d

def do_album_lyrics(l_g, alb: int, trax: dict):
    """
    do_album_lyrics aggregates all lyrics for a single album
    :param alb: lyric genius album key
    :param trax: dictionary accessed as track_dict[album id]['tracks']
    :return str of lyrics for all songs in album
    """
    gs_tracks = {}
    gs_albums = {}
    alb_lyrics: str = ""
    # iterate through albums then tracklists, get lyrics for each song id
    this_tracks = trax['tracks']
    z = len(this_tracks)
    for this_song in iter(this_tracks):
        song_dict = this_song['song']
        song_id = song_dict['id']
        song_title = song_dict['title']
        gs_tracks[song_id] = song_title
        try:
            this_lyrics = l_g.lyrics(song_id)
        except ConnectionResetError:
            try:
                this_lyrics = l_g.lyrics(song_id)
            except ConnectionResetError:
                this_lyrics = l_g.lyrics(song_id)
            else:
                alb_lyrics += str(this_lyrics)
        else:
            alb_lyrics += str(this_lyrics)
    gs_albums[alb] = gs_tracks

    return alb_lyrics, gs_albums

def get_lyrics(l_g, trax: dict):
    """
    get_lyrics returns a string of lyrics for single album
    :param l_g: lyric genius key
    :param trax: dictionary created by get_cleantracks
    :return str of lyrics
    """
    alb_lyrics: str = ""
    # iterate through albums then tracklists, get lyrics for each song id
    for song in iter(trax.keys()):
        this_lyrics: str = ""
        try:
            this_lyrics = l_g.lyrics(song)
        except ConnectionResetError:
            try:
                this_lyrics = l_g.lyrics(song)
            except ConnectionResetError:
                this_lyrics = l_g.lyrics(song)
        finally:
            if this_lyrics:
                alb_lyrics += str(this_lyrics)
            else:
                print("failed to add lyrics for %s" %(str(trax[song][1])))

    return alb_lyrics

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

def load_lyrics(afile: str):
    """
    loads a corpus of lyrics
    :param afile: str with filename
    :return list of str corpus
    """
    song_corp: list = []
    tmp: str = ""
    rd_h = open(afile, mode="r")
    for tmp in rd_h:
        song_corp.append(tmp)
    rd_h.close()

    return song_corp

def get_cleantracks(t_d: dict):
    """
    get_cleantracks creates a clean dict of songs for each of an artist's albums
    Args:
        t_d: dict of tracks from return value of get_albumtrax
    Returns: a_d
    """
    a_d = {}
    for alb_id in iter(t_d):
        al_lst = alb_id['tracks']
        trk_dict = {}
        for y in iter(al_lst):
            # al_lst is a list of len: num of tracks, each element a dict
            trk_num = y['number']
            title = y['song']['title']
            song_id = y['song']['id']
            song_url = y['song']['url']
            trk_dict[song_id] = trk_num, title, song_url
        a_d[alb_id] = trk_dict
        trk_dict = {}

    return a_d
