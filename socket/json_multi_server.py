from platform import java_ver
from json_socket import Server, Message
import json
import numpy as np
import pickle
import tensorflow as tf

class FLServer:
    EXP_UNIFORM = 0
    EXP_RANDOM_SAME_SIZE = 1
    EXP_RANDOM_DIFF_SIZE = 2
    EXP_SKEWED = 3

    def __init__(self, host, port, max_clients, max_round):
        self.host = host
        self.port = port
        self.server = Server(host, port, max_con=5)
        self.model = self.compile_model(self.build_model())

        self.max_clients = max_clients
        self.max_round = max_round

    def build_model(self):
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
        return model
    
    def prepare_dataset(self, name):
        if name == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
            
        if name == "cifar10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data(path="cifar10.npz")

        if name == "cifar100":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(path="cifar100.npz")
        
        if name == "imdb":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(path="imdb.npz")

        if name == "fmnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data(path="fmnist.npz")

    def split_dataset(self, y_train, experiment, num_samples):
        # Don't need x_train (which is big) for splitting the dataset
        size = len(y_train)
        train_idxs = {i:[] for i in range(size)}
        
        client_data_idxs = {i: [] for i in range(self.max_clients)}
        for i, v in enumerate(y_train):
            train_idxs[v].append(i)
        num_labels = len(train_idxs)
        if experiment == self.EXP_UNIFORM:
            for i in range(num_labels):
                indices = train_idxs[i]
                random_idxs = np.random.choice(indices, size=num_samples//num_labels, replace=True).tolist()
                client_data_idxs[i] = random_idxs

        if experiment == self.EXP_RANDOM_SAME_SIZE:
            random_indices = np.random.choice([i for i in range(size)], size=num_samples)

        


        
    def compile_model(self, model, optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def get_compile_config(self):
        config = {
            "optim": tf.keras.optimizers.serialize(self.optimizer), 
            "loss": tf.keras.losses.serialize(self.loss), 
            "metrics": self.metrics
        }
        return config
    
    def get_model_arch(self):
        arch = {
            "arch": self.model.to_json()
        }
        return arch
    
    def get_model_param(self):
        model_param = {
            "param": list(map(lambda layer: layer.tolist(), self.model.get_weights()))
        }
        return model_param

    def build_model_arch_msg(self):
        msg = Message(source=0, flag=Message.FLAG_GET_ARCH, data=self.get_model_arch())
        return msg

    def build_model_param_msg(self):
        msg = Message(source=0, flag=Message.FLAG_GET_PARAMS, data=self.get_model_param())
        return msg

    def build_compile_config_msg(self):
        msg
    def build_num_data_msg(self):
        msg = Message(source=0, flag=Message.FLAG_GET_NUM_DATA, data=self.get)
host = 'localhost'
port = 20000

server = Server(host, port, max_con=5)

client_counts = 2

current_round = 0
maximum_round = 5

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

optimizer = tf.keras.optimizers.Adam(lr=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = ['accuracy']
optim_config = tf.keras.optimizers.serialize(optimizer)
loss_config = tf.keras.losses.serialize(loss)
metrics_config = metrics
config = model.to_json()

while True:
    for i in range(client_counts):
        msg = Message(source=0, flag=Message.FLAG_GET_CURRENT_ROUND, data={"curr_round":current_round, "max_round":maximum_round}) # server msg source = 0  

        conn = server.accept(i)
        msg = conn.recv(i)
        
        flag = msg.flag
        if flag == Message.FLAG_GET_ARCH:
            msg = Message(source = i, flag=flag, data={"arch": list(map(lambda layer: layer.tolist(), model.get_weights()))})

            if len(msg) != 0:
                conn.send(i, msg)
        
        if flag == Message.FLAG_GET_CURRENT_ROUND:
            msg = Message(source = i, flag=flag, data={"curr_round": current_round})
            if len(msg) != 0:
                conn.send(i, msg)
        
        if flag == Message.FLAG_GET_ERROR_CODE:
            pass
    
        if flag == Message.FLAG_GET_LOSS:
            msg = 

        print(f"client {i}")
        print("flag", msg.flag)
        print("len", len(msg.data))

        if len(msg) != 0:
            conn.send(i, msg)

        else:
            conn.close(i)