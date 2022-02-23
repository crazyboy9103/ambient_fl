import copy
import numpy as np
import tensorflow as tf
import json
from app import numpy_encoder
import os
import random
import logging 
from datetime import datetime

class FederatedServer:
    client_number = 5 # 전체 클라이언트 개수
    server_weight = None # 현재 서버에 저장되어있는 weight
    local_weights = {} # 각 클라이언트에서 받아온 parameter들의 리스트

    experiment = 1 #Uniform by default

    done_clients = 0 # Task가 끝난 클라이언트의 개수
    server_round = 0 # 현재 라운드
    max_round = 5 #

    num_data = {}
    client_model_accuracy = {}
    server_model_accuracy = []

    model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(10, activation='softmax')
            ])

    
    optimizer = tf.keras.optimizers.SGD()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    logger = None 


    @classmethod
    def initialize(cls, client_num, experiment, max_round):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        cls.logger = cls.build_logger(current_time)
        cls.reset() # reset the variables when initialized
        cls.client_number = client_num
        cls.experiment = experiment
        cls.max_round = max_round
        cls.logger.INFO(f"Server initialized with {client_num} clients, experiment {experiment}, max round {max_round}")
        return "Initialized server"

    @classmethod
    def build_logger(cls, name):
        logger = logging.getLogger('log_custom')
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s;[%(levelname)s];%(message)s",
                              "%Y-%m-%d %H:%M:%S")
        
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(logging.INFO)
        logger.addHandler(streamHandler)

        fileHandler = logging.FileHandler(f'./log_{name}.txt' ,mode = "w")
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        
        logger.propagate = False
        return logger

    @classmethod
    def get_compile_config(cls):
        optim_config = cls.optimizer.serialize()
        loss_config = cls.loss.serialize()
        metrics_config = json.dumps(cls.metrics)
        compile_config = {"optim": optim_config, "loss": loss_config, "metrics":metrics_config}
        return compile_config

    @classmethod
    def get_model_as_json(cls):
        config = cls.model.to_json()
        return config


    @classmethod
    def update_num_data(cls, client_id, num_data):
        cls.num_data[client_id] = num_data
        cls.logger.INFO(f"Client {client_id} contains {num_data} data samples")
        return f"Number of data for {client_id} updated"


    @classmethod
    def update(cls, client_id, local_weight):
        if not local_weight:
            cls.logger.INFO(f"Client {client_id} weight error")
            print(f"Client {client_id} weight error")

            if client_id in cls.client_model_accuracy:
                cls.client_model_accuracy[client_id].append(0)

            else:
                cls.client_model_accuracy[client_id] = [0]


        else:
            cls.logger.INFO(f"Client {client_id} weight updated")
            print(f"Client {client_id} weight updated")

            local_param = list(map(lambda weight: np.array(weight, dtype=np.float32), local_weight))
            cls.local_weights[client_id] = local_param
            cls.evaluateClientModel(client_id, local_param)
        
        cls.done_clients += 1 # increment current count

        if cls.done_clients == cls.client_number:
            cls.logger.INFO(f"Round {cls.server_round} FedAvg with {cls.client_number} clients, experiment {cls.experiment}, max round {cls.max_round}, data samples {cls.num_data} ")
            cls.FedAvg() # fed avg
            cls.evaluateServerModel()
            cls.next_round()

        if cls.server_round == cls.max_round: # federated learning finished
            cls.logger.INFO("FL done")
            cls.save_ckpt() # save checkpoint
            cls.reset()

    @classmethod
    def FedAvg(cls):
        print("number", cls.client_number)
        print("exp", cls.experiment)
        print("done", cls.done_clients)
        print("server round", cls.server_round)
        print("max round", cls.max_round)
        print("num data", cls.num_data)

    
        print(f"FedAvg at round {cls.server_round}")
        weight = list(map(lambda block: np.zeros_like(block, dtype=np.float32), cls.local_weights[random.choice(list(cls.local_weights))]))

        total_num_data = 0
        for client_id in cls.local_weights:
            total_num_data += cls.num_data[client_id]

        for client_id, client_weight in cls.local_weights.items():
            client_num_data = cls.num_data[client_id]

            for i in range(len(weight)):
                weighted_weight = client_weight[i] * (client_num_data/total_num_data)
                weight[i] += weighted_weight
  
        cls.set_server_weight(weight)

    @classmethod
    def evaluateClientModel(cls, client_id, weight):
        cls.model.set_weights(cls.local_weights[client_id]) # change to local weight

        mnist = tf.keras.datasets.mnist
        (_, _), (test_images, test_labels) = mnist.load_data()
        n = len(test_images)
        indices = np.random.choice([i for i in range(n)], n//10)

        test_images = test_images[indices]
        test_labels = test_labels[indices]
        test_images = test_images / 255
        test_images = test_images.reshape(-1,28, 28, 1)

        acc = cls.model.evaluate(test_images, test_labels)[1] 

        if client_id not in cls.client_model_accuracy:
            cls.client_model_accuracy[client_id] = []

        cls.logger.INFO(f"Round {cls.server_round} Client {client_id} accuracy {acc}")
        cls.client_model_accuracy[client_id].append(acc)

        if cls.server_weight != None:
            cls.model.set_weights(cls.server_weight) # revert to server weight

    @classmethod
    def evaluateServerModel(cls):
        mnist = tf.keras.datasets.mnist
        (_, _), (test_images, test_labels) = mnist.load_data()
        n = len(test_images)
        indices = np.random.choice([i for i in range(n)], n//10)

        test_images = test_images[indices]
        test_labels = test_labels[indices]
        test_images = test_images / 255
        test_images = test_images.reshape(-1,28, 28, 1)

        acc = cls.model.evaluate(test_images, test_labels)[1] # first index corresponds to accuracy
        cls.logger.INFO(f"Round {cls.server_round} Server model accuracy {acc}")
        # each index corresponds to a round
        cls.server_model_accuracy.append(acc)

    @classmethod
    def next_round(cls):
        cls.done_clients = 0 # reset current
        cls.server_round += 1 # proceed
        cls.num_data = {}

    @classmethod
    def save_ckpt(cls):
        cls.model.save_weights("./checkpoints/FL")
        
    @classmethod
    def reset(cls):
        cls.client_model_accuracy = {}
        cls.server_model_accuracy = []
        cls.server_weight = None
        cls.local_weights = {}
        cls.done_clients = 0
        cls.server_round = 0
        cls.num_data = {}

    @classmethod
    def set_server_weight(cls, weight):
        cls.server_weight = weight

    @classmethod
    def get_server_weight(cls):
        return cls.server_weight

    @classmethod
    def get_done_clients(cls):
        return cls.done_clients

    @classmethod
    def get_server_round(cls):
        return cls.server_round
