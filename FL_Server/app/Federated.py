import copy
import numpy as np
import logging
logger = logging.getLogger(__name__)
import tensorflow as tf
class FederatedServer:
    client_number = 5 # 전체 클라이언트 개수
    global_weight = None # 현재 서버에 저장되어있는 weight
    local_weights = [] # 각 클라이언트에서 받아온 parameter들의 리스트
    
    experiment = 1 #Uniform by default
    
    current_count = 0 # Task가 끝난 클라이언트의 개수
    current_round = 0 # 현재 라운드
    
    fed_id = 0 # 각 클라이언트를 구별하기 위한 아이디
    total_num_data = 0 # 모든 클라이언트가 본 전체 데이터 개수
    
    accuracy = {}
    
    accuracies = {} # for all clients
    
    model = None
    
    
    
    @classmethod
    def __init__(cls):
        print("Federated init")
        cls.build_model()

    @classmethod
    def update(cls, local_weight):
        weight_list = []
      
        for i in range(len(local_weight)): 
            temp = np.array(local_weight[i])
            weight_list.append(temp)
            
        cls.current_count += 1 # increment current count
        cls.local_weights.append(weight_list) # append the received local weight to the local weights list that contains all other

        if cls.current_count == cls.client_number: # if current count = the max count
            cls.avg() # fed avg 
            cls.current_count = 0
            cls.current_round += 1
            accuracies[cls.current_round] = accuracy
            accuracy = {}
    
    @classmethod
    def build_model(cls):
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
    def avg(cls):
        # averages the parameters
        
        temp_weight = cls.local_weights.pop()
        for i in range(len(temp_weight)):
            temp_weight[i] = np.array(temp_weight[i])

        for i in range(len(cls.local_weights)):
            for j in range(len(cls.local_weights[i])):
                temp = np.array(cls.local_weights[i][j])
                temp_weight[j] = temp_weight[j] + temp
        
        cls.global_weight = temp_weight
        
        cls.model.set_weights(local_weight)
        
        mnist = tf.keras.datasets.mnist
        (_, _), (test_images, test_labels) = mnist.load_data()
        test_images = test_images / 255
        test_images = test_images.reshape(-1,28, 28, 1)
        
        acc = cls.model.evaluate(test_images, test_labels)
        accuracy[0] = acc
        
        cls.local_weights = []  # reset local weights
        cls.local_num_data = []
    
    @classmethod
    def reset(cls):
        cls.accuracies = {}
        cls.accuracy = {}
        cls.global_weight = None
        cls.local_weights = []
        cls.local_num_data = []
        cls.current_count = 0
        cls.current_round = 0
        cls.experiment = 1
        cls.fed_id = 0
        cls.total_num_data = 0
        #tentative
        tf.keras.backend.clear_session()
        cls.build_model()
    @classmethod
    def get_avg(cls):
        return cls.global_weight

    @classmethod
    def get_current_count(cls):
        return cls.current_count

    @classmethod
    def get_current_round(cls):
        return cls.current_round
