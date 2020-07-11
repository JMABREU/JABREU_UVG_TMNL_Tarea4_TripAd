
import gensim
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import CoherenceModel


def getLdaModel (pDictionaryId, pCorpus):
    vLdaModel = gensim.models.ldamodel.LdaModel(corpus=pCorpus,
                                                id2word=pDictionaryId,
                                                num_topics=20,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # Print the Keyword in the 10 topics
    print(vLdaModel.print_topics())
    return vLdaModel

def getLdaPerplexityModel(pLdaModel, pCorpus):
    #Perplejidad: a measure of how good the model is. lower the better.
    vPerplexity = pLdaModel.log_perplexity(pCorpus)

    return vPerplexity

def getLdaCoherenceModel(pLdaModel, pDictionaryId, pDataLematize):
    # Score de coherencia
    coherence_model_lda = CoherenceModel(model=pLdaModel, texts=pDataLematize, dictionary=pDictionaryId, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    return coherence_lda

def getVisualization(pLdaModel, pCorpus, pDictionaryId):
    # Visualizamos los temas
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(pLdaModel, pCorpus, pDictionaryId)
    return vis