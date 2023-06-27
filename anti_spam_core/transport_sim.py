import pandas as pd
import numpy as np
import pika


def get_channel():
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
    return channel

if __name__ == "__main__":
    data = pd.DataFrame()
    data = pd.read_csv("./tp.csv")
    item_id = data['id']
    text = data['text']
    
    output = pd.DataFrame(columns=['id', 'text'])
    output['id'] = item_id
    output['text'] = text
    output_json = output.to_json(orient='records')
    channel = get_channel()
    channel.basic_publish(exchange="news_exchange", routing_key="k1", body=output_json)    
