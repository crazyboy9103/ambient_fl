# Usage : python jetson_client.py --ip IP --p PORT
import matplotlib.pyplot as plt
import argparse
import json
import os
import threading
import time
from random import random
import numpy as np
import requests
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Quiet tensorflow error messages


class NumpyEncoder(json.JSONEncoder): # inherits JSONEncoder 
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

class Client:
    def __init__(self, max_round: int, time_delay = 5, suppress=True, num_samples=600):
        """
        @params: 
            experiment : Desired data split type (1~4)
            max_round : the maximum number of rounds that should be trained (arbitrary integer)
            model : the NN model type (either 'ann' or 'cnn')
            time_delay : the time delay until the next local check (arbitrary positive integer) 
                        (Need to increase this value if one round of training takes much longer than current time_delay. 
                        The reason is that any network communication until next round after the client has already uploaded 
                        the parameters for current round increases network overhead. Thus, higher time_delay will make communication
                        more stable while increasing the absolute time it takes. Requires careful selection of this value.)
            suppress : boolean value to print the logs
        
        @return: 
            None : Initializes the variables
                   Setup the urls for communication
                   Fetch client's id from the server
                   Downloads MNIST dataset and splits
                   Build model
        """
        base_url = f"http://{IP}:{PORT}/" # Base Url that we communicate with
        self.weight_url = base_url + "weight" # Url that we send or fetch weight parameters
        self.round_url = base_url + "round" # Url that helps synchronization
        self.id_url = base_url+"get_id" # Url from which we fetch the current client's id
        self.total_num_data_url = base_url + "total_num" # Url from which we fetch the number of total data points (seen by N clients)
        self.experiment_url = base_url + "experiment"
        self.accuracy_url = base_url + "accuracy"
        self.fed_id = self.request_fed_id() 
        self.experiment = self.request_experiment() # Experiment to test the performance of federated learning regime 
        
        self.time_delay = time_delay
        
        self.suppress = suppress
        '''
        Initial setup
        '''
        self.global_round = self.request_global_round()
        self.current_round = 0
        
        #self.change_client_number(max_round)
        self.max_round = max_round # Set the maximum number of rounds
        '''
        Downloads MNIST dataset and prepares (train_x, train_y), (test_x, test_y)
        '''
        self.train_images, self.train_labels = None, None
        self.test_images, self.test_labels = None, None
        self.prepare_images()
        
        self.train_index_list = None
        self.test_index_list = None
        self.split_train_images = []
        self.split_train_labels = []
        
        self.local_data_num = 0
        self.data_split(num_samples=num_samples)
        
        '''
        Builds model
        '''
        self.model = None
        self.build_cnn_model()
        
    def prepare_images(self):
        """
        @params: 
            model : 'ann' or 'cnn'. They need slightly different format for the input. For cnn, we add additional dimension for channel
        
        @return: 
            None : Prepares MNIST images in the required format for each model
            
        """
        mnist = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        self.train_images, self.test_images = self.train_images / 255, self.test_images / 255
        
        # For CNN, add dummy channel to feed the images to CNN
        self.train_images=self.train_images.reshape(-1,28, 28, 1)
        self.test_images=self.test_images.reshape(-1,28, 28, 1)
            
    
    def build_cnn_model(self):
        """
        @params: 
            None
        
        @return: 
            None : saves the CNN model in self.model variable 
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        
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
        if self.train_index_list is None or self.test_index_list is None:
            self.train_index_list = [[], [], [], [], [], [], [], [], [], []]
            self.test_index_list = [[], [], [], [], [], [], [], [], [], []]
            for i, v in enumerate(self.train_labels):
                self.train_index_list[v].append(i)

            for i, v in enumerate(self.test_labels):
                self.test_index_list[v].append(i)

        
        self.split_train_images = []
        self.split_train_labels = []
        
        if self.experiment == 1: #uniform data split
            self.local_data_num = num_samples
            


            for i in range(len(self.train_index_list)):
                indices = self.train_index_list[i]
                random_indices = np.random.choice(indices, size=num_samples//10)
                
                self.split_train_images.extend(self.train_images[random_indices])
                self.split_train_labels.extend(self.train_labels[random_indices])
            

        elif self.experiment == 2: # Randomly selected, equally sized dataset
            self.local_data_num = num_samples
            random_indices = np.random.choice([i for i in range(len(self.train_labels))], size=num_samples)
            self.split_train_images = self.train_images[random_indices]
            self.split_train_labels = self.train_labels[random_indices]

            counts = [0 for _ in range(10)]
            
            for label in self.train_labels[random_indices]:
                counts[label] += 1
            



        elif self.experiment == 3: # Randomly selected, differently sized dataset
            n = np.random.randint(1, num_samples)
            self.local_data_num = n
            random_indices = np.random.choice([i for i in range(len(self.train_labels))], size=n)
            self.split_train_images = self.train_images[random_indices]
            self.split_train_labels = self.train_labels[random_indices]
            

            counts = [0 for _ in range(10)]
            
            for label in self.train_labels[random_indices]:
                counts[label] += 1
            


            

        elif self.experiment == 4: #Skewed
            skewed_numbers = np.random.choice([i for i in range(10)], np.random.randint(1, 10))
            non_skewed_numbers = list(set([i for i in range(10)])-set(skewed_numbers))
            N = 0
            
            counts = [0 for _ in range(10)]
            
            for i in skewed_numbers:
                n = np.random.randint(50, 60)
                N += n
                
                indices = self.train_index_list[i]
                random_indices = np.random.choice(indices, size=n)
                
                self.split_train_images.extend(self.train_images[random_indices])
                self.split_train_labels.extend(self.train_labels[random_indices])
                
                counts[i] += n
            
                
            for i in non_skewed_numbers:
                n = np.random.randint(1, 10)
                N += n
                
                indices = self.train_index_list[i]
                random_indices = np.random.choice(indices, size=n)
                
                self.split_train_images.extend(self.train_images[random_indices])
                self.split_train_labels.extend(self.train_labels[random_indices])
                
                counts[i] += n
            
            
            
            self.local_data_num = N
        
        else:
            print("Pick from 1,2,3,4")
            return 
    
        self.split_train_images = np.array(self.split_train_images)
        self.split_train_labels = np.array(self.split_train_labels)
        
        self.update_total_num_data(self.local_data_num)    

        
        
    def update_total_num_data(self, num_data):
        """
        @params: 
            num_data : the number of training images that the current client has
        
        @return: 
            None : update the total number of training images that is stored in the server
        """
        local_num_data_to_json = json.dumps(num_data)
        requests.put(self.total_num_data_url, data=local_num_data_to_json)
    
    def request_total_num_data(self):
        """
        @params: 
            None
        
        @return: 
            result : Total number of training images available to all clients
        """
        result = requests.get(self.total_num_data_url)
        result = int(result.text)
        return result

    def request_fed_id(self):
        """
        @params: 
            None
        
        @return: 
            result : Automatically assigned client id that is given by the server
        """
        result = requests.get(self.id_url)
        result = result.json()
        return result
    
    def request_global_round(self):
        """
        @params: 
            None
        
        @return: 
            result : Current global round that the server is in
        """
        result = requests.get(self.round_url)
        result = result.json()
        return result
    
    def request_experiment(self):
        result = requests.get(self.experiment_url)
        result_data = result.json()
        
        if result_data is not None:
            return int(result_data)
        
        else:
            return 1
    
    def request_global_weight(self):
        """
        @params: 
            None
        
        @return: 
            global_weight : Up-to-date version of the model parameters
        """
        result = requests.get(self.weight_url)
        result_data = result.json()
        
        global_weight = None
        if result_data is not None:
            global_weight = []
            for i in range(len(result_data)):
                temp = np.array(result_data[i], dtype=np.float32)
                global_weight.append(temp)
            
        
        return global_weight

    def upload_local_weight(self, local_weight=[]):
        """
        @params: 
            local_weight : the local weight that current client has converged to
        
        @return: 
            None : Add current client's weights to the server (Server accumulates these from multiple clients and computes the global weight)
        """
        local_weight_to_json = json.dumps(local_weight, cls=NumpyEncoder)
        requests.put(self.weight_url, data=local_weight_to_json)
        
    def upload_local_accuracy(self, accuracy):
        temp_dict = {'acc':accuracy, 'id':self.fed_id}
        local_acc_to_json = json.dumps(temp_dict)
        requests.put(self.accuracy_url, data=local_acc_to_json)
        
    def validation(self, local_weight=[]):
        """
        @params: 
            local_weight : the current client's weights
        
        @return: 
            acc : test accuracy of the current client's model
        """
        if local_weight is not None:
            self.model.set_weights(local_weight)
            acc = self.model.evaluate(self.test_images, self.test_labels, verbose=0 if self.suppress else 1)
            self.upload_local_accuracy(acc)
            e = {out: acc[i] for i, out in enumerate(self.model.metrics_names)}

            return acc
        
    def train_local_model(self):
        """
        @params: 
            None
        
        @return: 
            local_weight : local weight of the current client after training
        """
        global_weight = self.request_global_weight()
        if global_weight != None:
            global_weight = np.array(global_weight)
            self.model.set_weights(global_weight)
        
        self.model.fit(self.split_train_images, self.split_train_labels, epochs=10, batch_size=16, verbose=0)
        N = self.request_total_num_data()
        
        local_weight = np.multiply(self.model.get_weights(), (self.local_data_num/N))
        return local_weight
    
    def task(self):
        """
        @params: 
            None
        
        @return: 
            None : Delayed execution of Federated Learning task
                  1. Check the client's current round
                      1.1. If the current round is 
        """
        
        #this is for executing on multiple devices
        self.global_round = self.request_global_round()

        if self.current_round >= self.max_round:
            print(f"Client {self.fed_id} finished")
            return 

        if self.global_round == self.current_round: #need update 
            global_weight = self.request_global_weight()

            local_weight = self.train_local_model()

            acc = self.validation(local_weight)

            self.upload_local_weight(local_weight)

            self.current_round += 1

            time.sleep(self.time_delay)

            return self.task()

        else: #need to wait until other clients finish
            time.sleep(self.time_delay * 2)
            return self.task()

        '''#this is for executing on multiple devices
        else:
            #this is for executing on one device
            self.global_round = self.request_global_round()
            


            if self.global_round == self.current_round: #need update 
                start = time.time()
                if not self.suppress:
                    print("Request global weight...")
                global_weight = self.request_global_weight()
                if not self.suppress:
                    print("Global weight request done")

                if not self.suppress:
                    print("Training local model...")
                local_weight = self.train_local_model()
                if not self.suppress:
                    print("Training done")

                acc = self.validation(local_weight)


                if not self.suppress:
                    print("Uploading local weight...")
                self.upload_local_weight(local_weight)
                if not self.suppress:
                    print("Weight upload done")

                if not self.suppress:
                    print("=========================")
                end = time.time()

                self.current_round += 1

                threading.Timer(self.time_delay, self.task, [multiple_devices]).start()

            else: #need to wait until other clients finish
                threading.Timer(self.time_delay*2, self.task, [multiple_devices]).start()
        #this is for executing on one device'''
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usage --ip {ip} --p {port} --max {max round} --delay {time delay} --num {num samples}")
    parser.add_argument("--ip", type=str, help="base ip address", default="127.0.0.1")
    parser.add_argument("--p", type=str, help="designated port", default="8000")
    parser.add_argument("--max", type=int, help="max round", default=5)
    parser.add_argument("--delay", type=int, help="time delay", default=5)
    parser.add_argument("--num", type=int, help="num samples", default=600)
    
    args = parser.parse_args()
    IP = args.ip
    PORT = args.p
    max_round = args.max
    time_delay = args.delay
    num_samples = args.num
    
    client = Client(max_round = max_round, time_delay = time_delay, num_samples)
    client.task()
