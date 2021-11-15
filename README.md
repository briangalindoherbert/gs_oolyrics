## musiclyrics_nlp
apply gensim vector training and LDA topic modeling to corpora of lyrics from selected musical artists

1. functions to interface with lyricgenius api to GET lyrics from Internet for an artist
2. parsing, scrubbing, and tokenization functions
3. uses gensim word2vec and Similarity for training of corpora to generate KeyedVector model
4. uses gensim.models.Ldamodel and gensim.corpora.Dictionary for LDA topic modeling

I initially built corpora of lyrics from complete 'official' album releases of The Grateful Dead and first three full-length releases from Kendrick Lamar.   As of this edit (Nov 14) I am working on full lyric corpus for a couple other artists. My target is to have lyrics from at least two artists per genre, and at least 3 genres.

In progress lyric corpora:
A. hip-hop:  have Kendrick Lamar, in process of adding Wu-Tang Clan and Outkast
B. 'hippie' rock- have Grateful Dead, working on adding Allman Brothers
C. alternative rock- building corpus from Nirvana, Pearl Jam, Green Day and Linkin Park

As of Nov 14, 2021 this 'app' is a rough draft, I have manually run various pieces and working on script cleanup 
and getting it all to hang together, by Thanksgiving weekend I will upload a seriously improved version!
