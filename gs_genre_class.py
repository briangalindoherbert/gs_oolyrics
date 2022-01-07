"""
file Docstring:
"""

from gs_datadict import GENRES, rapa, firstwavea, alternativea, countrya, metala, punka, \
    rocka, popa, folkrocka
from gs_utils import retrieve_genre_registry, clean_name


def clean_artlst_from_gen(genre: str):
    """
    returns list of clean artist names for a given genre (strip spaces, punc, extended chars)
    """
    genre_to_artists: dict = {
        'rap': rapa, 'firstwave': firstwavea, 'alternative': alternativea, 'country': countrya,
        'metal': metala, 'punk': punka, 'rock': rocka, 'pop': popa, 'folkrock': folkrocka
    }
    tmp_lst: list = genre_to_artists[genre]
    ctr: int = len(tmp_lst)
    for i in range(ctr):
        tmp_lst[i] = clean_name(tmp_lst[i])

    return tmp_lst

class Genre_Aggregator():
    """
    this Object is a container for musical genre information such as a list of all artists
    and artist track registries, for now.
    genre determines artists, and registries are retrieved by genre
    """
    def __init__(self, gen: str=''):
        self.trax: dict = {}
        if isinstance(gen, str):
            if gen == "all":
                self.genre: list = GENRES
                self.artist_list: list = []
                for genx in self.genre:
                    self.artist_list.extend(clean_artlst_from_gen(genre=genx))
            elif gen in GENRES:
                self.genre: str = gen
                self.artist_list: list = clean_artlst_from_gen(genre=gen)
        elif isinstance(gen, list):
            self.genre: list = []
            self.artist_list: list = []
            for genx in gen:
                if genx in GENRES:
                    self.genre.append(genx)
                    self.artist_list.extend(clean_artlst_from_gen(genre=genx))
        if not self.artist_list:
            raise KeyError("Valid Genre Not Received")
        if isinstance(self.genre, str):
            tmpgen = [self.genre]
        else:
            tmpgen = self.genre
        for genx in tmpgen:
            tmpreg: list = retrieve_genre_registry(genre=genx)
            for artreg in tmpreg:
                if isinstance(artreg, dict):
                    rootnam = str(list(artreg.keys())[0])[:-3]
                    self.trax[rootnam] = artreg

    def __iter__(self):
        """
        iterator for genre_aggregator is generator of artist names that object contains
        """
        for art, artreg in self.trax.items():
            for arttag, trakname in artreg.items():
                yield art, arttag, trakname

        return