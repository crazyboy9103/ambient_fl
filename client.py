import argparse
import json
import threading
import time
from random import random
import numpy as np
import requests
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Quiet tensorflow error messages

class Client:
    def __init__(self, ip_address, max_round, time_delay = 5, num_samples=600, client_id = 0, experiment = 1):
        # URLs
        self.session = requests.Session()
        self.base_url = ip_address
        self.update_num_data_url = self.base_url + "/update_num_data/" + str(client_id)
        self.put_weight_url =  self.base_url + "/put_local_weight/" + str(client_id)
        self.get_weight_url =  self.base_url + "/get_server_weight"
        self.round_url =  self.base_url + "/get_server_round" 
        self.get_model_url = self.base_url + "/get_server_model"
        self.get_compile_config_url = self.base_url + "/get_compile_config"
        
        # setting up variables
        self.experiment = experiment
        self.client_id = client_id
        self.time_delay = time_delay
        self.global_round = self.request_global_round()
        self.current_round = 0
        self.max_round = max_round # Set the maximum number of rounds
        
        # Download mnist dataset
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.prepare_images()
        
        
        # Splits dataset (iid, random, )
        self.split_train_images, self.split_train_labels = self.data_split(num_samples)
        self.local_data_num = len(self.split_train_labels)
        
        # Brings model architecture, optimizer, loss, metrics from the server  
        self.model = self.build_model_from_server()
        
        
        
    def prepare_images(self):
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images, test_images = train_images / 255, test_images / 255
        
        # For CNN, add dummy channel to feed the images to CNN
        train_images=train_images.reshape(-1,28, 28, 1)
        test_images=test_images.reshape(-1,28, 28, 1)
        return train_images, train_labels, test_images, test_labels
    
    def build_model_from_server(self):
        model = self.session.get(self.get_model_url).json()

        # None converted in null, which is unexpected..
        # Remap null to None
        model = tf.keras.models.model_from_json(model, custom_objects={"null":None}) 
        
        optimizer, loss, metrics = self.request_compile_config()
        model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        return model
        
        
    def data_split(self, num_samples):
        # Take a batch of num_samples according to self.experiment
        # 1 : iid 
        # 2 : Randomly selected, equally sized dataset
        # 3 : Randomly selected, differently sized dataset
        # 4 : Skewed dataset
        train_index_list = [[], [], [], [], [], [], [], [], [], []]
        test_index_list = [[], [], [], [], [], [], [], [], [], []]

        for i, v in enumerate(self.train_labels):
            train_index_list[v].append(i)

        for i, v in enumerate(self.test_labels):
            test_index_list[v].append(i)

        split_train_images = []
        split_train_labels = []

        if self.experiment == 1: #uniform data split
            # all 
            self.local_data_num = num_samples
            
            for i in range(len(train_index_list)):
                indices = train_index_list[i]
                random_indices = np.random.choice(indices, size=num_samples//10)
                
                split_train_images.extend(self.train_images[random_indices])
                split_train_labels.extend(self.train_labels[random_indices])
            

        elif self.experiment == 2: # Randomly selected, equally sized dataset
            self.local_data_num = num_samples
            random_indices = np.random.choice([i for i in range(len(self.train_labels))], size=num_samples)
            split_train_images = self.train_images[random_indices]
            split_train_labels = self.train_labels[random_indices]

        
            
        elif self.experiment == 3: # Randomly selected, differently sized dataset
            n = np.random.randint(1, num_samples)
            self.local_data_num = n
            random_indices = np.random.choice([i for i in range(len(self.train_labels))], size=n)
            split_train_images = self.train_images[random_indices]
            split_train_labels = self.train_labels[random_indices]
            
     
  
        elif self.experiment == 4: #Skewed
            all_labels = [i for i in range(10)]
            skewed_numbers = np.random.choice(all_labels, np.random.randint(1, 10))
            non_skewed_numbers = set(all_labels)-set(skewed_numbers)
            N = 0
          
            for i in skewed_numbers:
                n = np.random.randint(50, 60)
                N += n
                
                indices = train_index_list[i]
                random_indices = np.random.choice(indices, size=n)
                
                split_train_images.extend(self.train_images[random_indices])
                split_train_labels.extend(self.train_labels[random_indices])
                
                
            for i in non_skewed_numbers:
                n = np.random.randint(1, 10)
                N += n
                
                indices = train_index_list[i]
                random_indices = np.random.choice(indices, size=n)
                
                split_train_images.extend(self.train_images[random_indices])
                split_train_labels.extend(self.train_labels[random_indices])
      
            
            self.local_data_num = N
        
        # convert split dataset into numpy array
        split_train_images = np.array(split_train_images)
        split_train_labels = np.array(split_train_labels)
        return split_train_images, split_train_labels

        
        
    def update_total_num_data(self, num_data):
        # Upload client's num_data to the server 
        self.session.get(self.update_num_data_url +"/" + str(num_data))
        
    def request_compile_config(self):
        # Request optimizer, loss, metrics defined in the server
        compile_config = self.session.get(self.get_compile_config_url).json()
        optim = tf.keras.optimizers.deserialize(compile_config["optim"])
        loss = tf.keras.losses.deserialize(compile_config["loss"])
        metrics = compile_config["metrics"]
        return optim, loss, metrics
    
    def request_global_round(self):
        # Request the global round
        result = self.session.get(self.round_url)
        result = result.json()
        return result
    
    def request_global_weight(self):
        # Request the global model (parameters)
        result = self.session.get(self.get_weight_url)
        result_data = result.json()
        
        global_weight = None
        if result_data is not None:
            global_weight = []
            for i in range(len(result_data)):
                temp = np.array(result_data[i], dtype=np.float32)
                global_weight.append(temp)
            
        return global_weight

    def upload_local_weight(self, local_weight):
        # Upload trained local model parameters
        for i in range(len(local_weight)):
            local_weight[i] = local_weight[i].tolist()
        local_weight_to_json = json.dumps(local_weight)
        self.session.put(self.put_weight_url, data=local_weight_to_json)
        
    def train_local_model(self):
        # Train a local model from latest model parameters 
        print("train started")
        global_weight = self.request_global_weight()
        if global_weight != None:
            global_weight = list(map(lambda weight: np.array(weight), global_weight))
            self.model.set_weights(global_weight)
            
        
        self.model.fit(self.split_train_images, self.split_train_labels, epochs=10, batch_size=8, verbose=0)
        local_weight = self.model.get_weights()
        return local_weight

    def task(self):
        """
        Federated learning task
        1. If the current round is larger than the max round that we set, finish
        2. If the global round = current client's round, the client needs update
        3. Otherwise, we need to wait until other clients to finish
        """

        self.global_round = self.request_global_round()
        print(f"Client {self.client_id} current round {self.current_round} global round {self.global_round}")
        if self.current_round >= self.max_round: #need to terminate
            print(f"Client {self.client_id} finished")
            return 

        if self.global_round == self.current_round: #need to update 
            print(f"Client {str(self.client_id)} needs update")
            self.split_train_images, self.split_train_labels = self.data_split(num_samples=self.local_data_num)
            self.update_total_num_data(self.local_data_num) 
            self.current_round += 1
            local_weight = self.train_local_model()
            self.upload_local_weight(local_weight)
            time.sleep(self.time_delay)
            return self.task()

        else: #need to wait until other clients finish
            print(f"Client {self.client_id} needs wait")
            time.sleep(self.time_delay)
            return self.task()

"""if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, help="ip address of the server", default="http://147.47.200.178:9103/") 
    parser.add_argument("--round", '-r', type=int, help="max round", default=5)
    parser.add_argument("--num", '-n', type=int, help="number of samples (ignored if exp == 3, 4)", default=600)
    parser.add_argument("--id", type=int, help="client id", default=0)
    parser.add_argument("--exp", type=int, help="experiment number", default=1)
    parser.add_argument("--delay", type=int, help="time delay in seconds", default=5)
    args = parser.parse_args()
    
    client = Client(args.ip, args.round, args.delay, args.num, args.id, args.exp)
    client.task()"""
    
