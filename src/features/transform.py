from pandas import DataFrame

from src.data.prepare_data import fnConvertToList, fnCleanData, fnSentencesToWords
from src.features.dictionary import fnCreateDictionary, fnGenerateCorpus, fnGetCorpus
from src.features.utils import fnGetNGrams, fnRemoveStopwords, fnMakeBigrams, fnLemmatization
from src.models.train import getLdaModel, getLdaPerplexityModel, getLdaCoherenceModel

def setDataPrepare(pdf: DataFrame):

    data = fnConvertToList(pdf)
#    print(data[:1])

    data = fnCleanData(data)
#    print(data[:1])

    data_words = list(fnSentencesToWords(data))
    print(data_words[:1])

    return data_words

def setTransformData(pDataWords):
    # Construimos modelos de bigrams y trigrams
    (BigramsMod, TrigramsMod) = fnGetNGrams(pDataWords)
#    print(BigramsMod[pDataWords[0]])
#    print(TrigramsMod[BigramsMod[pDataWords[0]]])

    # Eliminamos stopwords
    data_words_nostops = fnRemoveStopwords(pDataWords, 'english')

    # Formamos bigrams
    data_words_bigrams = fnMakeBigrams(data_words_nostops, BigramsMod)

    # Lematizamos preservando únicamente noun, adj, vb, adv
    data_lemmatized = fnLemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print(data_lemmatized[:1])

    return data_lemmatized

def getLdaModelData(pData):

    # Creamos diccionario
    id2word = fnCreateDictionary(pData)
    print('Valores del Diccionario:')
    for key, value in id2word.items():
        print(key, value)

    # Create Corpus
    # Term Document Frequency
    corpus = fnGenerateCorpus(id2word, pData)
    # View
    print('Valores para Corpus:')
    #print(corpus[1:2])

    # Human readable format of corpus (term-frequency)
    fnGetCorpus(id2word, corpus)

    # Generate LDA Model
    lda_model = getLdaModel(id2word, corpus)
    print('Valores para lda_model:')
    print(lda_model)
    doc_lda = lda_model[corpus]
    print('Valores para doc_lda:')
    print(doc_lda)

    #Perplexity: a measure of how good the model is
    PerplexityModel = getLdaPerplexityModel(lda_model, corpus)
    print('Valor Perplexity: ', PerplexityModel)

    # Score de coherencia
    coherence_lda = getLdaCoherenceModel(lda_model, id2word, pData)
    print('Valor Coherence Score: ', coherence_lda)