import copy
import numpy as np
import tensorflow as tf
import json
from app import numpy_encoder
import os 
    
class FederatedServer:
    client_number = 5 # 전체 클라이언트 개수
    server_weight = None # 현재 서버에 저장되어있는 weight
    local_weights = {} # 각 클라이언트에서 받아온 parameter들의 리스트
    
    experiment = 1 #Uniform by default
    
    done_clients = 0 # Task가 끝난 클라이언트의 개수
    server_round = 0 # 현재 라운드
    max_round = 5 #
    total_num_data = 0 # 전체 데이터 개수
    
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
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
   
    @classmethod
    def initialize(cls, client_num, experiment, max_round):
        cls.client_number = client_num
        cls.experiment = experiment
        cls.max_round = max_round
        cls.client_model_accuracy = {}
        cls.reset() # reset the variables when initialized
        return "Initialized server"
    
    @classmethod
    def update_num_data(cls, client_id, num_data):
        cls.total_num_data += num_data
        cls.num_data[client_id] = num_data
        return f"Number of data for {client_id} updated"
    
    @classmethod
    def update(cls, client_id, local_weight):
        local_weight = list(map(lambda weight: np.array(weight, dtype=np.float32), local_weight))
        cls.local_weights[client_id] = local_weight
        cls.evaluateClientModel(client_id, local_weight) 
        cls.done_clients += 1 # increment current count
        
        if cls.done_clients == cls.client_number: 
            cls.FedAvg() # fed avg
            cls.evaluateServerModel()
            cls.next_round()
            cls.save() 
            
        if cls.server_round == cls.max_round: # federated learning finished
            cls.save() # save all history into json file 
            cls.reset()

    @classmethod
    def FedAvg(cls):
        """
        cls.local_weights contains key:value = client id:weight array
        
        - At this point, we do not know the shape of the weight array, so we use np.zeros_like function to make
        a temporary array filled with zeros, then accumulate the weights
        
        - The resulting weight must be a list of weights of type np.array
        
        - Fill in the blanks to implement FedAvg algorithm (just simple averaging)
        """ 
        ### TODO ###
        weight = list(map(lambda block: np.zeros_like(block, dtype=np.float32), cls.local_weights[0])) 
        # local weight와 같은 shape를 가지는 list<np.array> 를 만들기
        
        for client_id, client_weight in cls.local_weights.items():
            client_num_data = cls.num_data[client_id]

            for i in range(len(weight)):
                weighted_weight = client_weight[i] * (client_num_data/cls.total_num_data)
                weight[i] += weighted_weight
        ### TODO ###
        cls.set_server_weight(weight)
        #cls.evaluateServerModel()
        
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
        
        acc = cls.model.evaluate(test_images, test_labels)
        
        if client_id not in cls.client_model_accuracy:
            cls.client_model_accuracy[client_id] = []
       
        cls.client_model_accuracy[client_id].append(acc[1])
        
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
        # each index corresponds to a round
        cls.server_model_accuracy.append(acc) 
        
    @classmethod
    def next_round(cls):
        cls.done_clients = 0 # reset current
        cls.server_round += 1 # proceed
        cls.total_num_data = 0 # 전체 데이터 개수 
        cls.num_data = {} 
        
    @classmethod
    def save(cls):
        result = {"clients acc" : cls.client_model_accuracy, 
                  "server acc" : cls.server_model_accuracy}
        import json
        from time import gmtime, strftime
        timestamp = strftime("%Y%m%d_%H%M%S", gmtime())
        with open(f'{timestamp}.json', 'w') as f:
            json.dump(result, f)
           
        return f"Json file saved {timestamp}"

    @classmethod
    def reset(cls):
        cls.client_model_accuracy = {}
        cls.server_model_accuracy = []
        cls.server_weight = None
        cls.local_weights = {}
        cls.done_clients = 0
        cls.server_round = 0
        cls.num_data = {}
        cls.total_num_data = 0
    
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
