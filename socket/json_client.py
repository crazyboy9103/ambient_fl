from pickle_socket import Client, Message
import tensorflow as tf
import json
import pickle
import numpy as np

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
json_model = model.to_json()
params = model.get_weights()
params = list(map(lambda layer: layer.tolist(), params))

host = 'localhost'
port = 20000
msg = Message(source=0, flag=Message.FLAG_GET_PARAMS, data=params)
# Client code:
client0 = Client()
client0.connect(0, host, port)

client0.send(msg)
response = client0.recv()
print(response.get_current_round())
print(response.get_max_round())
print(response.get_id())
print(response.get_flag())
client0.close()


# client1 = Client()
# client1.connect(1, host, port)

# client1.send(config)
# response = client1.recv()
# #print("client1 gets", response)
# client1.close()