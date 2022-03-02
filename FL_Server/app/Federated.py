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
    # variables set by cls.initialize(client_num, experiment, max_round)
    client_number = 5 
    experiment = 1 
    max_round = 5 

    # variables reset by cls.reset()
    client_model_accuracy = {}
    server_model_accuracy = []
    server_weight = None # server's latest weight
    local_weights = {} # weights of each client
    done_clients = 0 # number of clients who finished training/uploading weights
    server_round = 0 # current round at the server
    num_data = {} 

    # model architecture
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

    # optimizer, loss, metrics 
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']

    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # logger
    logger = None 

    @classmethod
    def initialize(cls, client_num, experiment, max_round):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        cls.logger = cls.build_logger(current_time)
        cls.reset() # reset the variables when initialized
        cls.client_number = client_num
        cls.experiment = experiment
        cls.max_round = max_round
        cls.logger.info(f"Server initialized with {client_num} clients, experiment {experiment}, max round {max_round}")
        print(f"Log filename /home/ambient_fl/Logs/log_{current_time}.txt")
        return f"Log filename /home/ambient_fl/Logs/log_{current_time}.txt"

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

        fileHandler = logging.FileHandler(f'../Logs/log_{name}.txt', mode = "w")
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        
        logger.propagate = False
        return logger

    @classmethod
    def get_compile_config(cls):
        optim_config = tf.keras.optimizers.serialize(cls.optimizer)
        loss_config = tf.keras.losses.serialize(cls.loss)
        metrics_config = cls.metrics
        compile_config = {"optim": optim_config, "loss": loss_config, "metrics":metrics_config}
        return compile_config

    @classmethod
    def get_model_as_json(cls, **kwargs):
        config = cls.model.to_json(**kwargs)
        return config


    @classmethod
    def update_num_data(cls, client_id, num_data):
        cls.num_data[client_id] = num_data
        cls.logger.info(f"Client {client_id} contains {num_data} data samples")
        return f"Number of data for {client_id} updated"


    @classmethod
    def update_weight(cls, client_id, local_weight):
        if not local_weight:
            cls.logger.info(f"Client {client_id} weight error")

            if client_id in cls.client_model_accuracy:
                cls.client_model_accuracy[client_id].append(0)

            else:
                cls.client_model_accuracy[client_id] = [0]


        else:
            cls.logger.info(f"Client {client_id} weight updated")
            client_param = list(map(lambda weight: np.array(weight, dtype=np.float32), local_weight))
            cls.local_weights[client_id] = client_param
            cls.evaluateModel(client_id=client_id)
        
        cls.done_clients = len(cls.local_weights) # increment current count 

        if cls.done_clients == cls.client_number:
            cls.logger.info(f"Round {cls.server_round} FedAvg with {cls.client_number} clients, experiment {cls.experiment}, max round {cls.max_round}, data samples {cls.num_data} ")
            cls.FedAvg() # fed avg
            cls.evaluateModel(client_id=None) # Server's client_id is None
            cls.next_round()

        if cls.server_round == cls.max_round: # federated learning finished
            cls.logger.info("FL done")
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

        cls.server_weight = weight
       
    @classmethod
    def evaluateModel(cls, client_id = None):
        mnist = tf.keras.datasets.mnist
        (_, _), (test_images, test_labels) = mnist.load_data()
        n = len(test_images)
        indices = np.random.choice(n, n//10)

        test_images, test_labels = test_images[indices], test_labels[indices]
        test_images = test_images / 255
        test_images = test_images.reshape(-1,28, 28, 1)

        if not client_id: 
            acc = cls.model.evaluate(test_images, test_labels)[1] # first index corresponds to accuracy
            cls.logger.info(f"Round {cls.server_round} Server model accuracy {acc}")
            cls.server_model_accuracy.append(acc)

        else:
            cls.model.set_weights(cls.local_weights[client_id]) # change to local weight before evaluation
            acc = cls.model.evaluate(test_images, test_labels)[1] 

            if client_id not in cls.client_model_accuracy:
                cls.client_model_accuracy[client_id] = []

            cls.logger.info(f"Round {cls.server_round} Client {client_id} accuracy {acc}")
            cls.client_model_accuracy[client_id].append(acc)

            if cls.server_weight != None:
                cls.model.set_weights(cls.server_weight) # revert to server weight
   
    @classmethod
    def next_round(cls):
        cls.done_clients = 0 # reset current
        cls.server_round += 1 # proceed
        cls.num_data = {}

    @classmethod
    def save_ckpt(cls):
        cls.model.save_weights("../../checkpoints/FL")

    @classmethod
    def load_ckpt(cls, model_path):
        try:
            cls.model.load_weights(model_path)
        except:
            print(f"Failed to load from checkpoint {model_path}")
            cls.logger.info(f"Failed to load from checkpoint {model_path}")

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
    def get_server_round(cls):
        return cls.server_round
