from platform import java_ver
from pydoc import cli
from json_socket import Server, Message
import json
import numpy as np
import pickle
import tensorflow as tf
import time

class FLServer:
    EXP_UNIFORM = 0
    EXP_RANDOM_SAME_SIZE = 1
    EXP_RANDOM_DIFF_SIZE = 2
    EXP_SKEWED = 3

    def __init__(self, host, port, dataset_name):
        self.host = host
        self.port = port
        self.server = Server(host, port, max_con=5)
        self.model = self.compile_model(self.build_model())
        
        self.curr_round = 0

        self.dataset_name = dataset_name
        (_, self.y_train), (self.x_test, self.y_test) = self.prepare_dataset(dataset_name)


        
    def task(self):
        initialize
        train_once
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
            return tf.keras.datasets.mnist.load_data(path="mnist.npz")
            
        if name == "cifar10":
            return tf.keras.datasets.cifar10.load_data(path="cifar10.npz")

        if name == "cifar100":
            return tf.keras.datasets.cifar100.load_data(path="cifar100.npz")
        
        if name == "imdb":
            return tf.keras.datasets.imdb.load_data(path="imdb.npz")

        if name == "fmnist":
            return tf.keras.datasets.fashion_mnist.load_data(path="fmnist.npz")


    def split_dataset(self, experiment, num_samples):
        # Don't need x_train (which is big) for splitting the dataset
        size = len(self.y_train)
        train_idxs = {i:[] for i in range(size)}

        for i, v in enumerate(self.y_train):
            train_idxs[v].append(i)

        all_idxs = [i for i in range(len(self.y_train))]
        client_data_idxs = {i: [] for i in range(self.max_clients)}

        num_labels = len(train_idxs)
        if experiment == self.EXP_UNIFORM:
            for i in range(num_labels):
                indices = train_idxs[i]
                for client in client_data_idxs:
                    random_idxs = np.random.choice(indices, size=num_samples//num_labels, replace=True).tolist()
                    client_data_idxs[client].extend(random_idxs)
            
            return client_data_idxs

        if experiment == self.EXP_RANDOM_SAME_SIZE:
            for client in client_data_idxs:
                random_idxs = np.random.choice(all_idxs, size=num_samples).tolist()
                client_data_idxs[client].extend(random_idxs)

            return client_data_idxs
        
        if experiment == self.EXP_RANDOM_DIFF_SIZE:
            for i in range(num_labels):
                for client in client_data_idxs:
                    num_data_sample = np.random.randint(1, num_samples)
                    random_idxs = np.random.choice(all_idxs, size=num_data_sample).tolist()
                    client_data_idxs[client].extend(random_idxs)
            return client_data_idxs
        
        if experiment == self.EXP_SKEWED:
            all_labels = [i for i in range(num_labels)]
            skewed_labels = np.random.choice(all_labels, np.random.randint(1, num_labels))
            non_skewed_labels = set(all_labels)-set(skewed_labels)
            
            for i in skewed_labels:
                for client in client_data_idxs:
                    num_data = np.random.randint(int(0.1 * num_samples), num_samples)
                    indices = train_idxs[i]
                    random_idxs = np.random.choice(indices, size=num_data)
                    client_data_idxs[client].extend(random_idxs)
            
            for i in non_skewed_labels:
                for client in client_data_idxs:
                    num_data = np.random.randint(int(0.8 * num_samples), num_samples)
                    indices = train_idxs[i]
                    random_idxs = np.random.choice(indices, size=num_data)
                    client_data_idxs[client].extend(random_idxs)
           
            return client_data_idxs

    def request_status_code(self, id):
        msg = Message(source=-1, flag=Message.FLAG_GET_STATUS_CODE)
        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client
            return self.conns[id].recv(id)

    def request_terminate(self, id):
        msg = Message(source=-1, flag=Message.TERMINATE)
        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client
            
    def request_train(self, id):
        msg = Message(source=-1, flag=Message.FLAG_START_TRAIN, data={"start": True})
        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client
            return self.conns[id].recv(id)
            
    def send_dataset_name(self, id):
        msg = Message(source=-1, flag=Message.FLAG_GET_DATA_NAME, data={"dataset_name": self.dataset_name})
        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client

    def send_data_idxs(self, id):
        msg = Message(source=-1, flag=Message.FLAG_GET_DATA_IDX, data={"data_idxs": self.client_data_idxs[id]})
        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client
        
    def send_model_arch(self, id):
        msg = Message(source=-1, flag=Message.FLAG_GET_ARCH, data={"arch": self.model.to_json()})
        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client

    def send_model_param(self, id):
        msg = Message(source=-1, flag=Message.FLAG_GET_PARAMS, data={"param": list(map(lambda layer: layer.tolist(), self.model.get_weights()))})
        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client

    def send_compile_config(self, id):
        msg = Message(source=-1, flag=Message.FLAG_GET_CONFIG, data={
            "optim": tf.keras.optimizers.serialize(self.optimizer), 
            "loss": tf.keras.losses.serialize(self.loss), 
            "metrics": self.metrics
        })

        if len(msg) != 0:
            self.conns[id].send(id, msg) # uses connection with client and send msg to the client

    def initialize(self, experiment, num_samples, max_clients, max_round):
        self.client_data_idxs = self.split_dataset(experiment, num_samples)    
        self.conns = {i: self.server.accept(i) for i in range(max_clients)}
        self.max_round = max_round
        
        for id in range(max_clients):
            msg = self.request_status_code(id)
            source = msg.source # must be same with id 
            health = msg.data

            if health == Message.HEALTH_GOOD:
                self.send_dataset_name(source)
                self.send_data_idxs(source)
                self.send_model_arch(source)
                

            elif health == Message.HEALTH_BAD:
                self.request_terminate(source)
                self.conns[source].close()
                del self.conns[source]

        print(f"{len(self.conns)}/{max_clients} healthy")

    def train_once(self):
        # train and aggregate at once
        params = []
        for healthy_id in self.conns:
            msg = self.request_train(healthy_id)
            param = msg.data['param']
            self.params[healthy_id] = param
        
    def evaluate_param(self, id, param):
        if isinstance(param[0], list):
            param = list(map(lambda layer:np.array(layer), param))

        temp_param = self.model.get_weights()
        self.model.set_weights(param)
        self.model.evaluate()

    def reset(self):
        for client, conn in self.conns.items():
            conn.close()
        del self.conns
        self.conns = {}
        

    def compile_model(self, model, optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model
    
    
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
        

        print(f"client {i}")
        print("flag", msg.flag)
        print("len", len(msg.data))

        if len(msg) != 0:
            conn.send(i, msg)

        else:
            conn.close(i)