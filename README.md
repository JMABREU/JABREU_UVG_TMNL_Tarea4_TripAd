---------------------------------------------------------------------------------------------
UBICACIÓN CONTROL DE VERSIONES EN GITHUB
---------------------------------------------------------------------------------------------
https://github.com/JMABREU/JABREU_UVG_TMNL_Tarea4_TripAd


---------------------------------------------------------------------------------------------
COMANDOS APLICADOS AL AMBIENTE DE LA TAREA
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

---------------------------------------------------------------------------------------------
COMANDOS PARA APLICAR VERSIONAMIENTO EN GIT
---------------------------------------------------------------------------------------------
> git init
> git status
> git add api/*  (para agregar una carpeta completa al versionador con todos sus archivos)
> git rm api/test.py
> git rm --cached api/test.py	|  rm --cached api/__pycache__/*
> git add api/__init__.py api/main.py
> git add notebooks/bag_of_words.ipynb notebooks/prepare_data.ipynb notebooks/tokenize.ipynb notebooks/train.ipynb
> git status
> git add src/data/__init__.py src/data/prepare_data.py src/data/split.py
> git add src/features/__init__.py src/features/dictionary.py src/features/tokenize.py src/features/utils.py
> git add src/models/__init__.py src/models/predict.py src/models/train.py
> git add README.md requirements.txt
> git commit -m "Initial Sentiment Analysis Version Commit"

- Crear un nuevo proyecto
- main.py

from fastapi import FastApi

app = FastAPI()

@app.get('/{name}')
async def root(name: str):
	return {'message':f'hello{name}'}

- Procfile
	web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app
	web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

> conda create -n fastapi
> conda activate fastapi
(fastapi)> conda install python
(fastapi)> pip install -U fastapi uvicorn gunicorn
(fastapi)> pip freeze > requirements.txt


---------------------------------------------------------------------------------------------
COMANDOS PARA PUBLICAR EL GIT
---------------------------------------------------------------------------------------------
- Ingresar a Heroku.com
- Selecciona Python
- I'm ready to start
- Windows descargar e instalar
- (fastapi)> heroku login
- En la pagina inicial darle clic en Dashboard
- Ingresar en la opcion "Create New App"
- Colocarle un nombre a la app: jmabreu-uvg-tmnl-t4 y crear el app

- (fastapi)> git init
- (fastapi)> git add main.py Procfile requirements.txt
- (fastapi)> git commit -m "Initial Commit"
- (fastapi)> uvicorn main:app --reload
- (fastapi)> git init
- (fastapi)> heroku git:remote -a jmabreu-uvg-tmnl-t5
- (fastapi)> git add .
- (fastapi)> git commit -am "make it better"
- (fastapi)> git push heroku master
