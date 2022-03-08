from platform import java_ver
from pydoc import cli
from json_socket import Server, Message
import json
import numpy as np
import pickle
import tensorflow as tf
import time
import logging
from datetime import datetime

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



    def build_logger(self, name):
        logger = logging.getLogger('log_custom')
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s;[%(levelname)s];%(message)s",
                              "%Y-%m-%d %H:%M:%S")
        
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(logging.INFO)
        logger.addHandler(streamHandler)

        fileHandler = logging.FileHandler(f'log_{name}.txt', mode = "w")
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        
        logger.propagate = False
        return logger

    def task(self):
        if self.curr_round < self.max_round:
            self.curr_round += 1
            self.logger.info(f"Round {self.curr_round}/{self.max_round}")
        
        else:
            for id in self.conns:
                self.request_terminate(id)
            self.logger.info("Finished FL task")
            return 

        self.client_data_idxs = self.split_dataset(self.experiment, self.num_samples)  

        for id in self.client_data_idxs:
            self.send_data_idxs(id)



        
        self.logger.info("Started FL task")
        params, accs = self.train_once()
        
        for id in range(len(accs)):
           self.logger.info(f"client {id} acc {accs[id]}")

        N = sum(list(map(lambda idxs: len(idxs), self.client_data_idxs.values())))        
        
        aggr_layers = {}

        for id, param in params.items():
            n = len(self.client_data_idxs[id])
            self.logger.info(f"client {id}, {n} training data samples")
            for i, layer in enumerate(param):
                weighted_param = (n / N) * layer
                
                if i not in aggr_layers:
                    aggr_layers[i]  = []
                
                aggr_layers[i].append(weighted_param)
        
        weights = []

        for i, weighted_params in range(len(aggr_layers)):
            block = np.zeros_like(weighted_params[0], dtype=np.float32)
            for param in weighted_params:
                block += param

            weights.append(block)

        self.model.set_weights(weights)
        acc = self.evaluate_param(weights)
        self.logger.info(f"Server acc {acc}")
        

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
    
    def request_health_code(self, id):
        msg = Message(source=-1, flag=Message.FLAG_GET_STATUS_CODE)
        self.conns[id].send(id, msg)
        recv_msg = self.conns[id].recv(id)
        health = recv_msg.data
        return health

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
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.logger = self.build_logger(current_time)

        self.client_data_idxs = self.split_dataset(experiment, num_samples)    
        self.conns = {i: self.server.accept(i) for i in range(max_clients)}
        self.max_round = max_round
        self.experiment = experiment
        self.num_samples = num_samples

        for id in range(max_clients):
            health = self.request_health_code(id)

            if health == Message.HEALTH_GOOD:
                self.send_dataset_name(id)
                self.send_data_idxs(id)
                self.send_model_arch(id)
                self.send_compile_config(id)
                

            elif health == Message.HEALTH_BAD:
                self.request_terminate(id)
                self.conns[id].close()
                del self.conns[id]
                del self.client_data_idxs[id]

        print(f"{len(self.conns)}/{max_clients} healthy")
        return f"{len(self.conns)}/{max_clients} healthy"

    def train_once(self):
        # must run after initialize
        # train and aggregate at once
        params = {}
        accs = {} 

        for healthy_id in self.conns:
            msg = self.request_train(healthy_id)
            param = msg.data['param']
            param = list(map(lambda layer: np.array(layer), param))
            params[healthy_id] = param
            accs[healthy_id] = self.evaluate_param(param) # saves acc
        return params, accs

    def evaluate_param(self, param):
        if isinstance(param[0], list):
            param = list(map(lambda layer:np.array(layer), param))

        temp_param = self.model.get_weights()
        self.model.set_weights(param)

        n = len(self.x_test)
        idxs = np.random.choice(n, n//10, replace=False)
        x_test, y_test = self.x_test[idxs], self.y_test[idxs]

        acc = self.model.evaluate(x_test, y_test)[1]
        self.model.set_weights(temp_param)
        return acc
        

    def compile_model(self, model, optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

if __name__ == "__main__":
    experiment, num_samples, max_clients, max_round = 1, 200, 5, 5

    FL_Server = FLServer(host="127.0.0.1", port=20000, dataset_name="mnist") 
    print(FL_Server.initialize(experiment, num_samples, max_clients, max_round))
    FL_Server.task()