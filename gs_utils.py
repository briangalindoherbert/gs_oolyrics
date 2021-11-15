"""
gs_nlp_UTIL are my common utlity classes, REPLACE THIS WITH
SAME NAME SCRIPT IN GS_TWEET APP!!
"""
import re
import io
from nltk.tokenize import TweetTokenizer, TreebankWordTokenizer
from gs_datadict import engstoplist, STOP_LYRICS, GS_STOP
from wordcloud import WordCloud
from configmanager import Config, Item
import matplotlib as plt

def get_schema():
    """
    get_schema returns an instance of a configmanager Config for the twitter json data
    Returns: configmanager.Config instance
    """
    tw_schema = Config({"data": [{"text": Item(type=str, required=True),
                                  "id": Item(type=str, required=False),
                                  "created_at": Item(type=str, required=False),
                                  "author_id": Item(type=str, required=False),
                                  }],
                        "meta": {"newest_id": Item(type=str, required=False),
                                 "oldest_id": Item(type=str, required=False),
                                 "result_count": Item(type=int, required=False),
                                 "next_token": "b26v89c19zqg8o3fosnulspyeniqf4idkwtqiqrdfsgot"
                                 }
                        })
    return tw_schema

def get_long_schema():
    """
    get_schema returns an instance of a configmanager Config for the twitter json data
    Returns: configmanager.Config instance
    """
    long_schema = Config({"data": [{"text": Item(type=str, required=True),
                                    "id": Item(type=str, required=False),
                                    "created_at": Item(type=str, required=False),
                                    "author_id": Item(type=str, required=False),
                                    "attachments": {
                                        "media_keys": Item(type=str, required=False)
                                    },
                                    "entities": {
                                        "mentions": [{
                                            "start": Item(type=int, required=False),
                                            "end": Item(type=int, required=False),
                                            "username": Item(type=str, required=False)
                                        }]
                                    }
                                    }],
                          "meta": {"newest_id": Item(type=str, required=False),
                                   "oldest_id": Item(type=str, required=False),
                                   "result_count": Item(type=int, required=False),
                                   "next_token": "b26v89c19zqg8o3fosnulspyeniqf4idkwtqiqrdfsgot"
                                   }
                          })
    return long_schema

def get_raw_lyrics(lyf: str):
    """
    reads in a string of raw lyrics from a file
    :param lyf: str name of raw lyrics file
    :return:
    """
    with open(lyf, mode='r') as lyf:
        rawtext = lyf.read()

    return rawtext

def do_sent_tok(txt_str):
    """
    do_sent_tok uses either punkt sentence tokenizer or tweet tokenizer
    and it should pick up line breaks and periods \. for sentence breaks
    if needed, use the following to detect \n for sentence break:
    str(x).splitlines()
    """
    inst_twtok = TweetTokenizer()
    lwr_sen: list = inst_twtok(txt_str, language='english')
    return lwr_sen

def do_wrd_tok(senstr: list):
    """ do_wrd_tok tokenizes words from a list where each
    list item is a sentence
    """
    inst_wtok = TreebankWordTokenizer()
    wdata: list = []
    cleanlst: list = []
    for stx in iter(senstr):
        tmp_w: list = inst_wtok(stx)
        for iter_w in iter(tmp_w):
            if iter_w not in engstoplist:
                wdata.append(iter_w)
        cleanlst.append(wdata)
        wdata = []
    return cleanlst

def word_extraction(sentstr, stops):
    words = re.sub("[^\w]", " ", sentstr).split()
    cleaned_text = [w.lower() for w in words if w not in stops]
    new_s: str = ""
    for x in cleaned_text:
        new_s = new_s + " " + x
    return new_s

def do_cloud(batch_tw_wrds, opt_stops: str = None, maxwrd: int = 100):
    """
    wordcloud package options can be explored via '?wordcloud' (python- show docstring)
    background_color="white" - lighter background makes smaller words more legible,
    max_words= this can prevent over clutter, mask=shape the cloud to an image,
    stopwords=ad-hoc removal of unwanted words, contour_width=3,
    :param batch_tw_wrds: list of list of word tokens for tweets
    :param opt_stops: str var name for optional stop list
    :return:
    """
    import matplotlib.pyplot as plt

    cloud_text = io.StringIO(newline="")
    for tok in batch_tw_wrds:
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
                        stopwords=opt_stops,
                        min_font_size=10).generate(cloud_text.getvalue())

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcld)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    return

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

def save_artist_stuff(stuff_dct: dict, the_file: str):
    """
    save dicts of albums or songs
    :param stuff_dict: artist info in dict format
    :param the_file: name of save file
    :return 0 if save OK
    """
    with open(the_file, mode='wt+', newline=None) as w_info:
        for k, v in stuff_dct.items():
            w_info.write("%s : %v \n" %(str(k), v))
    w_info.close()

    return 0