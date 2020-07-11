from typing import List, Tuple

from gensim.corpora import Dictionary


def fnCreateDictionary(documents: List[List[str]]):
    return Dictionary(documents)

def fnGenerateCorpus(pDictionaryId, pTexts):
    vCorpus = [pDictionaryId.doc2bow(text) for text in pTexts]
    return vCorpus

def fnGetCorpus(pDictionaryId, pCorpus):
    # Human readable format of corpus (term-frequency)
    [[(pDictionaryId[id], freq) for id, freq in cp] for cp in pCorpus[:1]]

def term_document_matrix(documents: List[List[str]], dictionary: Dictionary) -> List[List[Tuple[int, int]]]:
    return [dictionary.doc2bow(text) for text in documents]

