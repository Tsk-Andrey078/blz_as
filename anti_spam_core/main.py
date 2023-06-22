from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from anti_spam_core.stopwordsdict import russian
import pandas as pd
import numpy as np
import pymorphy2
import json
import os
from anti_spam_core.prepare_x import PrepareDataX

def filtration(pr_df):
    
    #Train model
    print(pr_df)
    # Получение пути к текущему файлу скрипта
    script_path = os.path.abspath(__file__)

    # Получение пути к родительской папке скрипта (папка, содержащая скрипт)
    parent_folder = os.path.dirname(script_path)

    # Получение пути к папке "Номер 2" внутри родительской папки
    folder_2_path = os.path.join(parent_folder, "dataset")
    full_path = folder_2_path + "/posts_inst_ready.csv"
    df = get_dataframe(full_path)
    stop = russian
    x_train = df.loc[:, 'text'].values
    y_train = df.loc[:, 'spam'].values

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, stop_words=stop)

    lr_tfidf = Pipeline([('vect', tfidf), ('clf', MultinomialNB(alpha=0.5))])
    lr_tfidf.fit(x_train, y_train)



    #Get prediction

        #Prepare Data
    pr = PrepareDataX()
    pr_df = pr.delete_null(pr_df)
    pr_df['text'] = pr_df['text'].apply(pr.preprocessing)
    pr_df.drop_duplicates(subset='text')

        #Get data
    #y_pred = pr_df.loc[:, 'prediction'].astype(str)
    id_pred = pr_df.loc[:, 'id'].astype(str)
    x_pred = pr_df.loc[:, 'text'].astype(str)

        #Sending to model
    output = lr_tfidf.predict_proba(x_pred)
    output_classes = np.where(output[:, 1] >= 0.94, 1.0, 0.0)
    #out_df = pd.DataFrame({'text': x_pred, 'prediction': y_pred, 'spam': output_classes, 'spam_probability': list(output)}).to_csv("./test_inst_result_5.csv")
    out_df = pd.DataFrame({'id': id_pred, 'text': x_pred, 'spam': output_classes}).to_csv('./posts_result_3.csv')
    out_df = pd.DataFrame({'id': id_pred, 'text': x_pred, 'spam': output_classes}).to_json(orient='records')

    return out_df

def get_dataframe(url):
    df = pd.DataFrame()
    df = pd.read_csv(url)
    return df

