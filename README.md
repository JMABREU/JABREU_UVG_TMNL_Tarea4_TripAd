UBICACIÓN CONTROL DE VERSIONES EN GITHUB
https://github.com/JMABREU/JABREU_UVG_TMNL_Tarea4_TripAd


---------------------------------------------------------------------------------------------
-- COMANDOS APLICADOS AL AMBIENTE DE LA TAREA
---------------------------------------------------------------------------------------------
> conda create -n T5_PSA_JABREU
> conda activate T5_PSA_JABREU
> conda install python jupyter notebook
> pip install spacy
> pip install spacy-lookups-data
> python -m spacy download en_core_web_sm
> python -m spacy download es_core_news_sm
> pip install gensim pandas pyldavis
> conda install matplotlib
> pip install en-core-web-sm-abd	| spacy.load('en_core_web_sm')
> pip freeze > requirements.txt
> pip install -U fastapi uvicorn
> uvicorn api.test:app --reload