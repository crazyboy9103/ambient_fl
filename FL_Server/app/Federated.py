import copy
import numpy as np
import logging
logger = logging.getLogger(__name__)
import tensorflow as tf
import json

f = open("training_history.json", "r")
json_history = json.load(f)

curr_id = 0
if "last_id" in json_history:
    last_id = int(json_history["last_id"])
    curr_id = last_id + 1
else:
    json_history["last_id"] = curr_id

json_history[curr_id] = {}

print(f"The id for this training is {curr_id}. Use it to retrieve the recorded results") 

class FederatedServer:
    client_number = 6 # 전체 클라이언트 개수
    global_weight = None # 현재 서버에 저장되어있는 weight
    local_weights = {} # 각 클라이언트에서 받아온 parameter들의 리스트
    
    experiment = 1 #Uniform by default
    
    current_count = 0 # Task가 끝난 클라이언트의 개수
    current_round = 0 # 현재 라운드
    
    
    fed_id = 0 # 각 클라이언트를 구별하기 위한 아이디
    max_round = 5 #
    total_num_data = 0 # 모든 클라이언트가 본 전체 데이터 개수
    
    accuracy = {} # for each client 
    num_data = {} # for each client
    accuracies = {} # for all clients, for all rounds
    
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
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    @classmethod
    def __init__(cls):
        pass

    @classmethod
    def update(cls, fed_id, local_weight):
        weight_list = []
      
        for i in range(len(local_weight)): 
            temp = np.array(local_weight[i])
            weight_list.append(temp)
        
        cls.local_weights[fed_id] = weight_list
        cls.current_count += 1 # increment current count

        if cls.current_count == cls.client_number: # if current count = the max count
            cls.avg() # fed avg 
            cls.current_count = 0
            cls.current_round += 1
            
            json_history[curr_id][cls.current_round] = {'accuracy': cls.accuracy, 'local_weights': cls.local_weights, 'experiment':cls.experiment}
            with open("training_history.json", "w") as f:
                json.dump(json_history, f)
    
            cls.accuracy = {}
            cls.local_weights = {}
            cls.local_num_data = {}

        if cls.current_round == cls.max_round:
            json.dump(json_history, f)
            print(f"Training finished. Training information saved in 'training_history.json' with key {curr_id}")
            cls.reset()
            print("Server reset complete")


    @classmethod
    def avg(cls):
        # averages the parameters
        # the weights are already weighted at the client side
        # so simply add weights 
        N = cls.total_num_data

        temp_weight = np.zeros_like(cls.local_weights[0], dtype=np.float32)

        #for i in range(len(temp_weight)):
        #    temp_weight[i] = temp_weight[i], dtype=np.float32)

        for fed_id, local_weights in cls.local_weights.items():
            n = cls.num_data[fed_id]
            local_weights = np.array(local_weights, dtype=np.float32)
            weighted_weights = (n/N) * local_weights
            temp_weight = temp_weight + weighted_weights
            
        cls.global_weight = temp_weight
        cls.model.set_weights(cls.global_weight)
        
        mnist = tf.keras.datasets.mnist

        (_, _), (test_images, test_labels) = mnist.load_data()

        n = len(test_images)
        indices = np.random.choice([i for i in range(n)], n//10)
        
        test_images = test_images[indices]
        test_images = test_images / 255
        test_images = test_images.reshape(-1,28, 28, 1)
        
        acc = cls.model.evaluate(test_images, test_labels)
        
        cls.accuracy[cls.client_number] = acc[1] # last id + 1 for the federated learning model accuarcy 
    
    @classmethod
    def reset(cls):
        #json.dump(json_history, f)
        cls.accuracy = {}
        cls.global_weight = None
        cls.local_weights = {}
        cls.num_data = {}
        cls.current_count = 0
        cls.current_round = 0
        cls.experiment = 1
        cls.fed_id = 0
        cls.num_data = {}
        cls.total_num_data = 0
        #tentative
        #tf.keras.backend.clear_session()

        cls.model = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), 
                    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), 
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
                    tf.keras.layers.Dropout(0.25), 
                    tf.keras.layers.Flatten(), 
                    tf.keras.layers.Dense(128, activation='relu'), 
                    tf.keras.layers.Dropout(0.5), 
                    tf.keras.layers.Dense(10, activation='softmax')
            ])
        cls.model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

       
    @classmethod
    def get_avg(cls):
        return cls.global_weight

    @classmethod
    def get_current_count(cls):
        return cls.current_count

    @classmethod
    def get_current_round(cls):
        return cls.current_round
