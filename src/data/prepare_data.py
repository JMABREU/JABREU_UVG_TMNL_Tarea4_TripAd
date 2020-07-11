import re
from typing import Generator, List

from gensim.utils import simple_preprocess
from pandas import DataFrame


# Convertir a una lista
def fnConvertToList(df: DataFrame) -> List[str]:
    data = df.review.values.tolist()
#    print(data[:1])
    return data


#Permite limiar los valores basicos
def fnCleanData(pList:List) -> List:
    vData = pList
    vData = [re.sub(r'\S*@\S*\s?', '', sent) for sent in vData]     # Eliminar emails
    vData = [re.sub(r'\s+', ' ', sent) for sent in vData]           # Eliminar newlines
    vData = [re.sub(r"\'", "", sent) for sent in vData]             # Eliminar comillas
    #pprint(vData[:1])

    return vData

## normalizamos, aplica minusculas, matan acentos o signos puntuacion.
## crea el listado de palabras de las oraciones anteriores.
def fnSentencesToWords(sentences: List[str]) -> Generator:
    for sentence in sentences:
        # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True elimina la puntuación

#data_words = list(fnSentencesToWords(data))



