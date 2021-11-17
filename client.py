import argparse
import json
import threading
import time
from random import random
import numpy as np
import requests
import tensorflow as tf

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Quiet tensorflow error messages

class NumpyEncoder(json.JSONEncoder): # inherits JSONEncoder 
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

class Client:
    def __init__(self, max_round: int, time_delay = 5, suppress=True, num_samples=600, client_id = 0, experiment = 1):
        '''
        Urls
        '''
        self.base_url = "http://147.47.200.178:9103/" # Base Url
        self.put_weight_url =  self.base_url + "put_weight/" + str(client_id)
        self.get_weight_url =  self.base_url + "get_server_weight" # Url that we send or fetch weight parameters
        self.round_url =  self.base_url + "get_server_round" 

        '''
        Initial setup
        '''
        self.experiment = experiment
        self.client_id = client_id
        self.time_delay = time_delay
        self.suppress = suppress
        self.global_round = self.request_global_round()
        self.current_round = 0
        self.max_round = max_round # Set the maximum number of rounds
        
        '''
        Downloads MNIST dataset and prepares (train_x, train_y), (test_x, test_y)
        '''
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.prepare_images()
        self.split_train_images, self.split_train_labels = self.data_split(num_samples)
        self.local_data_num = len(self.split_train_labels)
        
        '''
        Builds model
        '''
        self.model = self.build_cnn_model()
        
    def prepare_images(self):
        """
        return: 
            None : Prepares MNIST images in the required format for each model
            
        """
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images, test_images = train_images / 255, test_images / 255
        
        # For CNN, add dummy channel to feed the images to CNN
        train_images=train_images.reshape(-1,28, 28, 1)
        test_images=test_images.reshape(-1,28, 28, 1)
        return train_images, train_labels, test_images, test_labels
    
    def build_cnn_model(self):
        """
        @params: 
            None
        
        @return: 
            None : saves the CNN model in self.model variable 
        """
        #This model definition must be same in the server (Federated.py)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model
        
    def data_split(self, num_samples):
        """
        @params: 
            num_samples : The number of sample images in each client. This value is used for equally
                          sized dataset
        
        @return: 
            None : Split the dataset depending on the self.experiment value
           
                If self.experiment is 1: Uniform data split: We take equal amount of data from each class (iid)
                If self.experiment is 2: Random data split1: We take equal amount of data, but not uniformly distributed across classes
                If self.experiment is 3: Random data split2: We take different amount of data and not uniformly distributed across classes
                If self.experiment is 4: Skewed: We take disproportionate amount of data for some classes
                        
        """
        
        train_index_list = [[], [], [], [], [], [], [], [], [], []]
        test_index_list = [[], [], [], [], [], [], [], [], [], []]
        for i, v in enumerate(self.train_labels):
            train_index_list[v].append(i)

        for i, v in enumerate(self.test_labels):
            test_index_list[v].append(i)

        
        split_train_images = []
        split_train_labels = []
        
        """
        Todo : split the data according to the instructions        
        """
        
        """
        For each experiment, you must
            1. save the total number of samples to self.local_data_num variable
            2. add the split images and labels into self.split_train_images and self.split_train_labels respectively
        """
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
            temp = [i for i in range(10)]
            skewed_numbers = np.random.choice(temp, np.random.randint(1, 10))
            non_skewed_numbers = list(set(temp)-set(skewed_numbers))
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
        
        split_train_images = np.array(split_train_images)
        split_train_labels = np.array(split_train_labels)
        return split_train_images, split_train_labels

        
        
    def update_total_num_data(self, num_data):
        """
        num_data : the number of training images that the current client has
        
        update the total number of training images that is stored in the server
        """
        update_num_data_url =  self.base_url + "update_num_data/"+str(self.client_id)+"/"+str(num_data)
        requests.get(update_num_data_url)
        

    
    def request_global_round(self):
        """
        result : Current global round that the server is in
        """
        result = requests.get(self.round_url)
        result = result.json()
        return result
    
    def request_global_weight(self):
        """
        global_weight : Up-to-date version of the model parameters
        """
        result = requests.get(self.get_weight_url)
        result_data = result.json()
        
        global_weight = None
        if result_data is not None:
            global_weight = []
            for i in range(len(result_data)):
                temp = np.array(result_data[i], dtype=np.float32)
                global_weight.append(temp)
            
        return global_weight

    def upload_local_weight(self, local_weight):
        """
        local_weight : the local weight that current client has converged to
        
        Add current client's weights to the server (Server accumulates these from multiple clients and computes the global weight)
        """
        local_weight_to_json = json.dumps(local_weight, cls=NumpyEncoder)
        requests.put(self.put_weight_url, data=local_weight_to_json)
        
    def train_local_model(self):
        """
        local_weight : local weight of the current client after training
        """
        global_weight = self.request_global_weight()
        if global_weight != None:
            global_weight = np.array(global_weight)
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
        
        #this is for executing on multiple devices
        self.global_round = self.request_global_round()
        
        print("global round", self.global_round)
        print("current round", self.current_round)
        if self.current_round >= self.max_round:
            print(f"Client {self.client_id} finished")
            return 

        if self.global_round == self.current_round: #need update 
            print("Client "+ str(self.client_id) + "needs update")
            self.split_train_images, self.split_train_labels = self.data_split(num_samples=self.local_data_num)
            self.update_total_num_data(self.local_data_num)        
            local_weight = self.train_local_model()
            self.upload_local_weight(local_weight)
            self.current_round += 1
            time.sleep(self.time_delay)
            return self.task()

        else: #need to wait until other clients finish
            print("need wait")
            time.sleep(self.time_delay)
            return self.task()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", '-r', type=int, help="max round")
    parser.add_argument("--num", '-n', type=int, help="number of samples (overridden if exp == 3, 4")
    parser.add_argument("--id", type=int, help="client id")
    parser.add_argument("--exp", type=int, help="experiment number")
    args = parser.parse_args()
    
    client = Client(args.round, 5, True, args.num, args.id, args.exp)
    client.task()
    