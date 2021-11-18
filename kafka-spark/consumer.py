from kafka import KafkaConsumer
from json import loads

consumer = KafkaConsumer(
    'testtopic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: loads(x.decode('utf-8'))

)
# client = MongoClient('localhost:27017')
# collection = client.testtopic.testtopic
for message in consumer:
    message = message.value
    # message['timestamp'] = message['timestamp'][0]
    # message['vec'] = message['vec'][0][:10]
    print(message)
    # collection.insert_one(message)
    # print('{} added to {}'.format(message, collection))
