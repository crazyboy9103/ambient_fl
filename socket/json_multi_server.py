from platform import java_ver
from pydoc import cli
from json_socket import Server, Message, FLAGS
import numpy as np
import tensorflow as tf
import time
import logging
from datetime import datetime
import threading


class FLServer:
    EXP_UNIFORM = 0
    EXP_RANDOM_SAME_SIZE = 1
    EXP_RANDOM_DIFF_SIZE = 2
    EXP_SKEWED = 3

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = Server(host, port, max_con=5)
        self.curr_round = 0
        

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
        # Check round
        if self.curr_round < self.max_round:
            self.curr_round += 1
            self.logger.info(f"Round {self.curr_round}/{self.max_round}")
            print(f"Round {self.curr_round}/{self.max_round}")
        
        else:
            for id in self.server.clients:
                self.request_terminate(id)
            self.logger.info("Finished FL task")
            print("Finished FL task")
            return 

        # Split dataset 
        self.client_data_idxs = self.split_dataset(self.experiment, self.num_samples)  
        self.logger.info("Started FL task")
        print("Started FL task")
        
        # Gets trained parameters, accuracies
        params, accs = self.train_once(self.epochs, self.batch_size)
        
        # record accs in log
        for id in range(len(accs)):
            self.logger.info(f"client {id} acc {accs[id]}")
            print(f"client {id} acc {accs[id]}")
        # FedAvg Algorithm
        N = sum(map(lambda idxs: len(idxs), self.client_data_idxs.values()))       
        aggr_layers = {}

        for id, param in params.items():
            n = len(self.client_data_idxs[id])
            self.logger.info(f"client {id}, {n} training data samples")
            print(f"client {id}, {n} training data samples")
            for i, layer in enumerate(param):
                weighted_param = (n / N) * layer
                
                if i not in aggr_layers:
                    aggr_layers[i]  = []
                
                aggr_layers[i].append(weighted_param)
        
        print(aggr_layers)
        weights = []

        for i, weighted_params in aggr_layers.items():
            block = np.zeros_like(weighted_params[0], dtype=np.float32)
            for param in weighted_params:
                block += param

            weights.append(block)
        
        # swap & evaluate & record server model parameter
        self.model.set_weights(weights)
        acc = self.evaluate_param(weights)
        self.logger.info(f"Server acc {acc}")
        print(f"Server acc {acc}")

        return self.task()
        

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

        all_idxs = [id for id in range(len(self.y_train))]
        client_data_idxs = {id: [] for id in self.server.clients}

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


    def request_terminate(self, id):
        msg = Message(source=-1, flag=FLAGS.TERMINATE)
        self.server.send(id, msg) # uses connection with client and send msg to the client
        
            
    def request_train(self, id, epochs, batch_size):
        msg = Message(source=-1, flag=FLAGS.FLAG_START_TRAIN, data={
            "epochs": epochs, 
            "batch_size": batch_size,
            "data_idxs": self.client_data_idxs[id], 
            "param": list(map(lambda layer: layer.tolist(), self.model.get_weights()))
        })
        assert len(msg) != 0, "Message must contain data"
        self.server.send(id, msg) # uses connection with client and send msg to the client
        recv_msg = self.server.recv(id)
        param = recv_msg.data
        print("request train")
        return param


    def connect(self, id):
        print(f"accepting {id}")
        self.server.accept(id)
        print(f"client {id} registered")
        #self.conns[id] = self.server.clients[id]

    def initialize(self, dataset_name, experiment, num_samples, max_clients, max_round, epochs, batch_size):
        # prepare dataset
        self.dataset_name = dataset_name
        print("Prep dataset")
        (_, self.y_train), (self.x_test, self.y_test) = self.prepare_dataset(dataset_name)
        self.x_test = self.x_test.reshape(-1, 28, 28, 1)
        print("..done")

        # build & compile model 
        print("Build model")
        self.model = self.compile_model(self.build_model())
        print("..done")

        
        # setup logger
        print("Build logger")
        current_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.logger = self.build_logger(current_time)
        print("..done")
        
        
        # setup connections
        print("Build conns")
        #self.client_data_idxs = self.split_dataset(experiment, num_samples) 
        
        self.threads = []
        for i in range(max_clients):
            thread = threading.Thread(target = self.connect, args=(i,))
            self.threads.append(thread)
            thread.start()

        for thread in self.threads:
            thread.join()

        print("..done")
        print(len(self.server.clients))

        # variables
        print("Setting up variables")
        self.max_round = max_round
        self.experiment = experiment
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = batch_size
        print("..done")

        print("Health check")
        for id in range(max_clients):
            result_code = self.request_setup(id)
            print(result_code)
            healthy = result_code == FLAGS.RESULT_OK
            if healthy:
                self.logger.info(f"client {id} healthy")

            else:
                self.logger.info(f"client {id} not healthy")
                self.request_terminate(id)
                self.server.close(id)
                
        print("..done")
        assert len(self.server.clients) != 0, "no available clients"
        
        print(f"{len(self.server.clients)}/{max_clients} healthy")
        return f"{len(self.server.clients)}/{max_clients} healthy"

    def request_setup(self, id):
        msg = Message(source=-1, flag=FLAGS.FLAG_SETUP, data={
            "dataset_name": self.dataset_name, 
            "arch": self.model.to_json(),  
            "optim": tf.keras.optimizers.serialize(self.optimizer), 
            "loss": tf.keras.losses.serialize(self.loss), 
            "metrics": self.metrics
        })
        assert len(msg) != 0, "Message must contain data"
        self.server.send(id, msg)
        recv_msg = self.server.recv(id)
        print("source", recv_msg.source)
        print("data", recv_msg.data)
        result_code = recv_msg.data
        return result_code
            


    def train_once(self, epochs, batch_size):
        # must run after initialize
        # train and aggregate at once
        params = {}
        accs = {} 

        threads = []
        for healthy_id in self.server.clients:
            thread = threading.Thread(target = self.request_train, args=(healthy_id, epochs, batch_size, ))
            param = self.request_train(id=healthy_id, epochs=epochs, batch_size=batch_size)
            param = list(map(lambda layer: np.array(layer), param))
            params[healthy_id] = param
            accs[healthy_id] = self.evaluate_param(param) # saves acc
        return params, accs

    def evaluate_param(self, param):
        temp_param = self.model.get_weights()
        self.model.set_weights(param)

        n = len(self.x_test)
        idxs = np.random.choice(n, n//10, replace=False)
        x_test, y_test = self.x_test[idxs], self.y_test[idxs]

        acc = self.model.evaluate(x_test, y_test)[1]
        self.model.set_weights(temp_param)
        return acc
    
    #def run_on_thread(self, func):


    def compile_model(self, model, optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"]):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="experiment 1, 2, 3, 4", type=int, default=1)
    parser.add_argument("--num", help="num_samples", type=int, default=200)
    parser.add_argument("--cli", help="max_clients", type=int, default=1)
    parser.add_argument("--round", help="max_round", type=int, default=5)
    parser.add_argument("--data", help="dataset_name", type=str, default="mnist")
    parser.add_argument("--host", help="host", type=str, default="127.0.0.1")
    parser.add_argument("--port", help="port", type=int, default=20000)
    parser.add_argument("--epochs", help="epochs", type=int, default=10)
    parser.add_argument("--batch", help="batch_size", type=int, default=8)

    args = parser.parse_args()

    FL_Server = FLServer(host=args.host, port=args.port) 
    print(FL_Server.initialize(args.data, args.exp, args.num, args.cli, args.round, args.epochs, args.batch))
    FL_Server.task()