from fastapi import FastAPI, Query
import pandas as pd
from src.features.transform import *

app = FastAPI()


@app.post('/train')
async def train_model():
    #    train()

    # INICIO PROCESO PRINCIPAL

    df = pd.read_csv('data/raw/reviews.csv')
    print(df.head())
    print('Load File Completed')
    print('**************************************************************')

    dataWords = setDataPrepare(df)
    print('Data Preparation Completed')
    print('**************************************************************')

    dataTransform = setTransformData(dataWords)
    print('Data Transformation Completed')
    print('**************************************************************')

    getLdaModelData(dataTransform)
    print('Load Model Completed')
    print('**************************************************************')

    # FINALIZA EL PROCESO PRINCIPAL

    return {'Result': 'End Main'}
