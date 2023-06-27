from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from stopwordsdict import russian
import pandas as pd
import numpy as np
import pymorphy2
import json
import os
import pika
import time
import joblib
from prepare_modules.prepare_x import PrepareDataX

def filtration(pr_df):
    pr = PrepareDataX()

    #Get Model
    model = joblib.load(open("./models/as_model_inst.pkl", 'rb'))

    if len(pr_df) > 1:
        #Get prediction

            #Prepare Data    
        pr_df = pr.delete_null(pr_df)
        pr_df['text'] = pr_df['text'].apply(pr.preprocessing)
        pr_df.drop_duplicates(subset='text')

            #Get data
        #y_pred = pr_df.loc[:, 'prediction'].astype(str)
        id_pred = pr_df.loc[:, 'id'].astype(str)
        x_pred = pr_df.loc[:, 'text'].astype(str)

            

            #Sending to model
        output = model.predict_proba(x_pred)
        output_classes = np.where(output[:, 1] >= 0.94, 1.0, 0.0)
        
        out_df = pd.DataFrame({'id': id_pred, 'text': x_pred, 'spam': output_classes}).to_json(orient='records')

        return out_df
    else:
        print(pr_df)


def get_dataframe(url):
    df = pd.DataFrame()
    df = pd.read_csv(url)
    return df

def callback(ch, method, properties, body):
    data = pd.read_json(body.decode('utf-8'))
    output = filtration(data)
    ch.basic_publish(exchange="news_exchange", routing_key="k2", body=output)

if __name__ == "__main__":
    username = 'admin'
    password = 'admin'
    my_queue = 'news_queue'
    my_routing_key = "k1"
    

    # Open a connection to RabbitMQ on localhost using all default parameters
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host="rabbitmq", port=5672,
            credentials=pika.PlainCredentials(
                username=username,
                password=password
            ),
        ),
    )
    channel = connection.channel()
    channel.queue_declare(
        queue=my_queue,
        durable=True,
        exclusive=False,
        auto_delete=False,
    )
    channel.basic_consume(queue=my_queue, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()