# encoding=utf-8
"""
gs_datadict: contains reference data, such as file schemas, which are used
by gsLyrics.
"""
ARTISTDIR: str = "/Users/bgh/dev/pydev/gsLyrics/artists/"
LYRICDIR: str = "/Users/bgh/dev/pydev/gsLyrics/lyrics/"
RAWDIR: str = "/Users/bgh/dev/pydev/gsLyrics/rawdata/"
OUTDIR = '/Users/bgh/dev/pydev/gsLyrics/output/'
MODELDIR = '/Users/bgh/dev/pydev/gsLyrics/models/'
NLTKDIR = '/Users/bgh/dev/NLP/nltk_data'
W2VEC_PRE = '/Users/bgh/dev/pydev/superleague/models/freebase-vectors-skipgram1000-en.bin'
TW2VEC_PRE = '/Users/bgh/dev/pydev/superleague/models/word2vec_twitter_tokens.bin'

GENRES = ['rap', 'rock', 'firstwave', 'alternative', 'metal', 'punk', 'country', 'pop', 'folkrock']

rapa = ['Kendrick Lamar', 'Lil Jon', 'OutKast', '2Pac', 'TechN9ne', 'Yelawolf',
        'Snoop Dogg', 'Kanye West', 'Wu-Tang Clan', 'Jeezy', '50 Cent', 'Run-DMC',
        'Migos', 'Drake', 'Eminem', 'Wiz Khalifa']

firstwavea = ['The Cure', 'David Bowie', 'Siouxsie and the Banshees', 'Violent Femmes', 'Pretenders',
             'The Police', 'Elvis Costello', 'Tears for Fears', 'XTC', 'Eurythmics', 'Depeche Mode',
             'The Psychedelic Furs', 'The Smiths', 'The Beat', 'Oingo Boingo', 'INXS',
             'They Might Be Giants', 'Devo', 'REM', 'Joe Jackson', 'Radiohead', 'Blondie',
             'Yazoo','The Clash', 'Madness', 'Talking Heads', 'A Flock of Seagulls']

alternativea = ['Soundgarden', 'Linkin Park', 'Garbage', 'Green Day', 'Red Hot Chili Peppers',
               'Pearl Jam', 'Nirvana', 'Collective Soul', 'U2', 'Keane', 'Alice in Chains', 'Counting Crows']

countrya = ['The Chicks', 'Brad Paisley', 'Garth Brooks', 'Tim McGraw', 'George Strait',
            'JohnnyCash', 'Keith Urban', 'Dierks Bentley', 'Zac Brown Band', 'Alan Jackson']

metala = ["Metallica", "AC/DC", "Van Halen", "Guns N’ Roses", 'Mötley Crüe', 'Ozzy Osbourne',
         'Judas Priest', 'Black Sabbath', 'Iron Maiden', 'Five Finger Death Punch']

punka = ['The Velvet Underground', 'Boomtown Rats', 'Black Flag', 'Circle Jerks', 'Patti Smith',
         'Sex Pistols', 'Dead Kennedys', 'Ramones', 'blink-182', 'MDC', 'Lou Reed', 'Beat Farmers']

rocka = ['BadCompany', 'JoanJett', 'AllmanBrothersBand', 'CreedenceClearwaterRevival',
         'Eagles', 'BlackCrowes', 'TheJimiHendrixExperience', 'TheWho',
         'Santana', 'BruceSpringsteen', 'Rush', 'LynyrdSkynyrd',
         'TheGratefulDead', 'PinkFloyd', 'TomPetty', 'LedZeppelin', 'PatBenatar',
         'RollingStones', 'BluesTraveler', 'RyanAdams', 'TheBeatles',
         'LittleFeat']

popa = ['Train', 'Maroon5', '10000Maniacs', 'EdSheeran', 'Coldplay',
       'BlackEyedPeas', 'FleetwoodMac', 'BrunoMars', 'SarahMcLachlan', 'OneRepublic']

folkrocka = ['BobDylan', 'CaroleKing', 'JohnHiatt', 'NeilYoung', 'JohnMellencamp',
             'DaveMatthewsBand', 'Wilco']

# sentences:  Yelawolf, Kendrick Lamar, TechN9ne, LilJon, OutKast
RAP_SENTENCES = [['he', 'got', 'old', 'mossberg', 'in', 'mossy', 'oak', 'duffle', 'bag',
               'laying', 'in', 'back', 'dump', 'boy', 'do', 'not', 'make', 'go', 'pop', 'trunk'],
              ["walked", "back", "kentucky", "fried", "chicken", "light", "skinned",
               "nigga", "gap", "teeth", 'southside', 'projects', 'chiraq', 'terror', 'dome',
               'drove', 'california', 'woman', 'him', 'five-hundred', 'dollars', 'had',
               'son', 'hopin', 'that', 'he', 'would', 'see', 'college'],
              ['i', 'am', 'angel', 'this', 'wicked', 'planet', 'nobody', 'understands',
               'my', 'angle', 'love', 'i', 'was', 'sent', 'from', 'above', 'i', 'have',
               'landed', 'in', 'blood', 'psycho', 'bandit', 'i', 'have', 'become', 'frightful',
               'damage', 'scuffed'],
              ['patróns', 'rocks', 'i', 'ready', 'shots', 'women', 'come', 'around', 'every',
               'time', 'i', 'pour', 'shots', 'cups', 'air', 'everybody', 'take', 'shots', 'you',
               'feeling', 'drunk', 'put', 'your', 'hands', 'air', 'women', 'come', 'around',
               'every', 'time', 'i', 'pouring', 'shots'],
              ['she', 'had', 'fish', 'fries', 'cookouts', 'my', 'childs', 'birthday', 'i', 'aint',
               'invited', 'despite', 'i', 'show', 'her', 'utmost', 'respect', 'when', 'i', 'fall',
               'through', 'all', 'you', 'defend', 'lady', 'when', 'i', 'call', 'you', 'yeah', 'i',
               'sorry', 'ms', 'jackson', 'i', 'real', 'never', 'meant', 'make', 'your', 'daughter',
               'cry', 'i', 'apologize', 'trillion', 'times', 'i', 'sorry', 'ms', 'jackson', 'i',
               'real', 'never', 'meant', 'make', 'your', 'daughter', 'cry']
              ]

CHRISTMAS_SENTENCES = [['meadow', 'we', 'build', 'snowman', 'we', 'pretend', 'he', 'parson', 'brown',
                     'he', 'say', 'you', 'married', 'we', 'say', 'no', 'man', 'you', 'job', 'when',
                     'you', 'town', 'he', 'sings', 'love', 'song', 'we', 'go', 'along', 'walking',
                     'winter', 'wonderland'],
                    ['rudolph', 'red-nosed', 'reindeer', 'had', 'very', 'shiny', 'nose', 'you',
                     'saw', 'you', 'even', 'say', 'glows', 'all', 'other', 'reindeer', 'used',
                     'laugh', 'call', 'him', 'names', 'never', 'poor', 'rudolph', 'join',
                     'reindeer', 'games', 'one', 'foggy', 'christmas', 'eve', 'santa', 'came',
                     'say', 'rudolph', 'your', 'nose', 'bright', 'not', 'you', 'guide', 'my',
                     'sleigh', 'tonight'],
                     ]

# 1. George Strait, 2. George Strait, 3. Garth Brooks
COUNTRY_SENTENCES = [['took', 'my', 'saddle', 'houston', 'broke', 'my', 'leg', 'santa', 'fe', 'lost',
                      'my', 'wife', 'girlfriend', 'somewhere', 'along', 'i', 'looking', 'eight',
                      'when', 'pull', 'gate', 'i', 'hope', 'judge', 'aint', 'blind', 'amarillo',
                      'morning', 'amarillos', 'my', 'mind'],
                     ['cowboys', 'sure', 'fun', 'racing', 'wind', 'chasing', 'sun', 'i', 'take',
                      'time', 'time', 'those', 'crazy', 'friends', 'mine', 'head', 'out', 'steel',
                      'horses', 'wheels', 'we', 'ride', 'we', 'burn', 'up', 'road', 'old', 'mexico',
                      'blend', 'desert', 'we', 'amigos', 'we', 'roll'],
                     ['blame', 'all', 'my', 'roots', 'i', 'showed', 'up', 'boots', 'ruined', 'your',
                      'black', 'tie', 'affair', 'i', 'saw', 'surprise', 'fear', 'eyes', 'when', 'i',
                      'took', 'glass', 'champagne', 'i', 'toasted', 'you', 'said', 'honey', 'we',
                      'through', 'you', 'never', 'hear', 'complain', 'i', 'friends', 'low', 'places',
                      'where', 'whiskey', 'drowns', 'beer', 'chases', 'my', 'blues', 'away']
                     ]

FIRSTW_SENTENCES = [['pretty', 'women', 'out', 'walking', 'gorillas', 'down', 'my', 'street',
                    'from', 'my', 'window', 'i', 'am', 'staring', 'my', 'coffee', 'goes', 'cold',
                    'look', 'over', 'there', 'where', 'there', 'there', 'lady', 'that', 'i',
                    'used', 'know', 'she', 'married', 'now', 'engaged', 'something', 'i', 'am', 'told'],
                    ['i', 'guess', 'got', 'something', 'do', 'luck', 'i', 'waited', 'my', 'whole',
                     'life', 'one', 'day', 'after', 'day', 'i', 'get', 'angry', 'i', 'will', 'say',
                     'that', 'day', 'in', 'my', 'sight', 'when', 'i', 'take', 'bow', 'say', 'goodnight',
                     'mo', 'my', 'momma', 'momma', 'mo', 'my', 'mum', 'have', 'you', 'kept', 'your',
                     'eye', 'your', 'eye', 'your', 'son', 'i', 'know', 'you', 'have', 'had', 'problems',
                     'you', 'are', 'not', 'only', 'one', 'when', 'your', 'sugar', 'left', 'he',
                     'left', 'you', 'run']
                    ]

rock_artist_ids = {12712: 'Nirvana', 22696: 'Pearl Jam', 21900: "The Grateful Dead", }
rap_artist_ids = {1421: "Kendrick Lamar", 108: "50 Cent"}
PEARL_JAM_DCT = {19251: "Ten", 24232: "Vs.", 24273: "Vitalogy", 24233: "No Code", 514052: "Given to Fly",
             86180: "Yield", 100210: "Binaural", 65729: "Riot Act", 26329: "Lost Dogs", 154552: "Rearviewmirror",
             100461: "Pearl Jam", 64338: "Backspacer", 39251: "Lightning Bolt", 605789: "MTV Unplugged",
             593889: "Gigaton"
             }
DEAD_DCT = {105519: "Workingman's Dead", 105547: "In the Dark", 105860: "Dead Set",
            105546: "Reckoning", 105548: "Go to Heaven", 26596: "Shakedown Street",
            25528: "Terrapin Station", 41113: "Steal Your Face", 105530: "Blues for Allah",
            105527: "From the Mars Hotel", 105544: "Wake of the Flood",
            105526: "History of the Grateful Dead (Bear's Choice)", 48231: "Europe '72",
            26446: "Skull and Roses", 18482: "American Beauty", 18650: "Aoxomoxoa",
            105514: "Live / Dead", 18576: "Anthem of the Sun", 18484: "The Grateful Dead",
            588045: "The Very Best of Grateful Dead", 663866: "Dave's Picks Volume 29",
            508738: "Red Rocks Amphitheatre 1978", 508706: "The Best of the Grateful Dead"
            }
REM_DCT = {91073: "Fables of the Reconstruction", 559711: "Out of Time", 91154: "Monster",
           91065: "Reckoning", 91060: "Murmur", 221701: "Eponymous", 394496: "Green",
           91580: "Life's Rich Pageant", 81206: "Document"}
CORESTOPS = ["a", "about", "also", "am", "among", "an", "and", "any", "are", "as",
             "at", "be", "because", "but", "by", "can", "cause", "could", "do",
             "either", "else", "ever", "for", "from", "got", "ha", "have", "his",
             "hers", "how", "however", "if", "is", "it", "just", "let",
             "like", "likely", "may", "me", "might", "must", "of", "off",
             "often", "oh", "on", "or", "should", "so", "some", "than", "that",
             "the", "their", "then", "there", "these", "they", "this", "tis",
             "to", "too", "twas", "us", "was", "way", "what",
             "which", "while", "whom", "will", "with", "would", "yall", "yet",
             ]

# translate converts chars to their numeric representation, as in ord(charstr)
QUOTEDASH_TABLE = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–－᠆–-", "'''''-----")])
CHAR_CONVERSION = {
    u"\u200B": "",  # this one's a bugger- 'zero-length space' unicode- aka invisible!
    u"\u2002": " ",
    u"\u2003": " ",
    u"\u2004": " ",
    u"\u2005": " ",
    u"\u2006": " ",
    u"\u2010": "-",
    u"\u2011": "-",
    u"\u2012": "-",
    u"\u2013": "-",
    u"\u2014": "-",
    u"\u2015": "-",
    u"\u2018": "'",
    u"\u2019": "'",
    u"\u201a": "'",
    u"\u201b": "'",
    u"\u201c": "'",
    u"\u201d": "'",
    u"\u201e": "'",
    u"\u201f": "'",
    u"\u2026": "'",
    "－": "-",
    u"\u00f6": "o",         # this and next inspired by Motley Crue
    u"\u00fc": "u",
}

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

GS_CONTRACT = {
    r"ain't": "aint",
    r"aren't": "are not",
    r"arent": "are not",
    r"can't": "can not",
    r"cant": "can not",
    r"'cause": "because",
    r"citys": "city is",
    r"could've": r"could have",
    r"couldn't": r"could not",
    r"didn't": r"did not",
    r"doesn't": r"does not",
    r"don't": r"do not",
    r"dont": r"do not",
    r"'em": r"them",
    r"everything's": r"everything is",
    r"hadn't": r"had not",
    r"hasn't": r"has not",
    r"haven't": r"have not",
    r"he'd": r"he would",
    r"he'll": r"he will",
    r"he's": r"he is",
    r"how'd": r"how did",
    r"how'll": r"how will",
    r"how's": r"how is",
    r"i'd": r"i would",
    r"i'll": r"i will",
    r"i'm": r"i am",
    r"i've": r"i have",
    r"isn't": r"is not",
    r"isnt": r"is not",
    r"it'd": r"it would",
    r"it'll": r"it will",
    r"it'll've": r"it will have",
    r"its": r"it is",
    r"it's": r"it is",
    r"let's": r"let us",
    r"ma'am": r"mam",
    r"mayn't": r"may not",
    r"might've": r"might have",
    r"mightn't": r"might not",
    r"mo'": "more",
    r"must've": r"must have",
    r"mustn't": r"must not",
    r"needn't": r"need not",
    r"o'clock": r"oclock",
    r"oughtn't": r"ought not",
    r"shan't": r"shall not",
    r"she'd": r"she would",
    r"she'll": r"she will",
    r"she's": r"she is",
    r"should've": r"should have",
    r"shouldn't": r"should not",
    r"so've": r"so have",
    r"so's": r"so as",
    r"that'd": r"that would",
    r"that's": r"that is",
    r"there'd": r"there would",
    r"there's": r"there is",
    r"they'd": r"they would",
    r"they'll": r"they will",
    r"theyll": r"they will",
    r"they're": r"they are",
    r"theyre": r"they are",
    r"they've": r"they have",
    r"wasn't": r"was not",
    r"we'd": r"we would",
    r"we'll": r"we will",
    r"we're": r"we are",
    r"we've": r"we have",
    r"weren't": r"were not",
    r"what'll": r"what will",
    r"what're": r"what are",
    r"what's": r"what is",
    r"whats": r"what is",
    r"what've": r"what have",
    r"when's": r"when is",
    r"whens": r"when is",
    r"when've": r"when have",
    r"where'd": r"where did",
    r"where's": r"where is",
    r"who'll": r"who will",
    r"who's": r"who is",
    r"whos r": r"who is r",
    r"who've": r"who have",
    r"why's": r"why is",
    r"won't": r"will not",
    r"wont": r"will not",
    r"would've": r"would have",
    r"wouldn't": r"would not",
    r"y'all": r"yall",
    r"you'd": r"you would",
    r"youd": r"you would",
    r"you'll": r"you will",
    r"you're": r"you are",
    r"you've": r"you have"
}
