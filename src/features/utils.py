
from typing import List

import gensim
import nltk
import spacy
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords


# Construimos modelos de bigrams y trigrams
# https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
def fnGetNGrams(pDataWords):
    vBigram = gensim.models.Phrases(pDataWords, min_count=5, threshold=100)
    vTrigram = gensim.models.Phrases(vBigram[pDataWords], threshold=100)

    # Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
    vBigraMod = gensim.models.phrases.Phraser(vBigram)
    vTrigraMod = gensim.models.phrases.Phraser(vTrigram)

    #print(bigram_mod[data_words[0]])
    #print(trigram_mod[bigram_mod[data_words[0]]])

    return (vBigraMod, vTrigraMod)



# Eliminar stopwords
def fnRemoveStopwords(documents, pType) -> List[List[str]]:
    nltk.download('stopwords')
    stop_words = stopwords.words(pType)
    if pType=='english':
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in documents]


# Hacer bigrams
def fnMakeBigrams(pTexts, pBigraMod):
    return [pBigraMod[doc] for doc in pTexts]

# Hacer trigrams
def fnMakeTrigrams(pTexts, pBigraMod, pTrigraMod):
    return [pTrigraMod[pBigraMod[doc]] for doc in pTexts]



def fnLemmatization(pTexts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load("en_core_web_sm")
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in pTexts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Lematización basada en el modelo de POS de Spacy
# def fnLemmatization(pTexts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in pTexts:
#         doc = (" ".join(sent))
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out
