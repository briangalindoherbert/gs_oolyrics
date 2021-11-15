# encoding=utf-8
"""
gs_datadict: contains reference data, such as file schemas, which are used
by gsSentiment.
"""
from configmanager import Config, Item, ConfigError, Section
from configparser import ConfigParser

OUTDIR = '/Users/bgh/dev/pydev/gsSentiment/output/'
MODELDIR = '/Users/bgh/dev/pydev/superleague/models/'
W2VEC_PRE = '/Users/bgh/dev/pydev/superleague/models/freebase-vectors-skipgram1000-en.bin'
TW2VEC_PRE = '/Users/bgh/dev/pydev/superleague/models/word2vec_twitter_tokens.bin'

parser = ConfigParser()
parser.read_dict({'section1': {'key1': 'value1',
                               'key2': 'value2',
                               'key3': 'value3'},

                  'section2': {'keyA': 'valueA',
                               'keyB': 'valueB',
                               'keyC': 'valueC'}
                  })
engstoplist = ["a", "all", "almost", "also", "among", "an", "and", "any",
               "as", "at", "be", "because", "been", "but", "by", "cause", "could",
               "dear", "either", "else", "ever", "every", "for", "from", "had", "have", "his",
               "hers", "how", "however", "if", "in", "into", "it", "just", "least", "let",
               "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no",
               "nor", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather",
               "says", "should", "since", "so", "some", "than", "that", "the", "their", "them",
               "then", "there", "these", "this", "those", "tis", "to", "too", "twas", "us", "wants",
               "what", "when", "where", "which", "while", "who", "whom", "why", "with", "yet", "your"]

NL = '\n'
SP = ' '
RPT1 = '['
RPT2 = ']'
SCT1 = '{'
SCT2 = '}'

GS_ADVB = ["am", "are", "as", "be", "been", "being", "do", "does", "doing", "did",
           "have", "has", "had", "having", "is", "until", "was", "were"]
GS_PPRON = ["we", "he", "her", "him", "me", "she", "them", "us", "they"]
GS_STOP = ["RT", "a", "about", "all", "almost", "also", "among", "am", "an", "and", "already",
           "any", "are", "as", "at", "back", "because", "but", "by", "came", "cause", "come",
           "could", "day", "did", "dont", "does", "earlier", "either", "else", "ever", "even",
           "for", "follow", "from", "go", "going", "got", "has", "had", "have", "his", "her",
           "hers", "how", "not", "however", "if", "I", "in", "into", "is", "it", "its",
           "just", "least", "who", "let", "lets", "like", "likely", "may", "me",
           "might", "must", "much", "my", "need", "no", "now", "of", "often", "on",
           "one", "only", "or", "other", "our", "past",
           "own", "part", "rather", "really", "same", "seems", "sent", "shall", "show",
           "should", "since", "so", "some", "something", 'still', 'such', "than", "that",
           "thats", "the", "their", "them", "then", "there", "these", "they", "think",
           "this", "those", "thus", "to", "too", "two", "used", "was", "watch", "well",
           "were", "what", "while", "who", "will", "would", "whom",
           "with", "yet", "your", "rt", "we", "what", "been", "more", "when",
           "big", "after", "he", "man", "us", "off", "les", "des", "et",
           "il", "en", "before", "di", "us", "very", "you"]
STOP_LYRICS = ['aint', 'am', 'are', 'back', 'came', 'can', 'cannot', 'cant', 'come', 'dont',
         'everything', 'get', 'getting', 'go', 'going', 'gonna', 'got', 'gotta', 'has', 'he',
         'her', 'him', 'i', 'id', 'ill', 'im', 'is', 'isnt', 'ive', 'keep', 'know',
         'la', 'let', 'made', 'maybe', 'need', 'never', 'now', 'not', 'oh',
         'ooh', 'one', 'really', 'said', 'say', 'see', 'she', 'shes', 'still',
         'sure', 'take', 'tell', 'thats', 'theres', 'they', 'theyre', 'thing', 'think',
         'time', 'told', 'two', 'walk', 'whats', 'was', 'way', 'we', 'want', 'well',
         'went', 'were', 'whatll', 'will', 'wont', 'word', 'would', 'you',
         'make', 'nothing', 'give', 'hear', 'right', 'night', 'day', 'little', 'sometimes']

IDIOM_MODS = {'darth vader': -2.5, 'male privilege': -2.5, "good guys": 0.5}
VADER_MODS = {"amf": -2.0, "sociopathic": -2.5, "cartel": -1.0, "ideologues": -0.5,
              "blunder": -0.5, "commodotize": -0.5
              }
