import json
from django.http import HttpResponse
# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from app import numpy_encoder
from app.Federated import FederatedServer

import logging
logger = logging.getLogger(__name__)

@api_view(['GET'])
def index(request):
    logger.info("request index")
    return HttpResponse("index ok", status.HTTP_200_OK)

@api_view(['GET'])
def round(request):
    return HttpResponse(FederatedServer.get_current_round(), status.HTTP_200_OK)
    
@api_view(['GET', 'PUT'])
def experiment(request):
    if request.method == 'GET':
        json_data = json.dumps(FederatedServer.experiment)
        return HttpResponse(json_data, status.HTTP_200_OK)
            
    elif request.method == "PUT":
        json_data = JSONParser().parse(request)
        FederatedServer.experiment = int(json_data)
        return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['PUT'])
def accuracy(request):
    if request.method == 'PUT':
        json_data = JSONParser().parse(request)
        FederatedServer.accuracy[int(json_data['fed_id'])] = float(json_data['accuracy'])
        return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['PUT'])
def client_num(request):
    json_data = JSONParser().parse(request)
    FederatedServer.client_number = int(json_data)
    return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['GET', 'PUT'])
def weight(request):
    if request.method == 'GET':
        global_weight = FederatedServer.get_avg()
        global_weight_to_json = json.dumps(global_weight, cls=numpy_encoder.NumpyEncoder)
        return HttpResponse(global_weight_to_json, status.HTTP_200_OK)

    elif request.method == 'PUT':
        json_data = JSONParser().parse(request)
        fed_id = json_data['fed_id']
        weights = json_data['weights']
        FederatedServer.update(fed_id, weights)
        return HttpResponse("Request PUT OK", status.HTTP_200_OK)

    else:
        return HttpResponse("Request OK", status.HTTP_200_OK)

@api_view(['GET', 'PUT'])
def total_num_data(request):
    if request.method == 'GET':
        return HttpResponse(FederatedServer.total_num_data, status.HTTP_200_OK)

    elif request.method == 'PUT':
        json_data = JSONParser().parse(request)
        fed_id, num_data = int(json_data["fed_id"]), int(json_data["num_data"])
        FederatedServer.total_num_data += num_data
        FederatedServer.num_data[fed_id] = num_data
        return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['GET'])
def get_id(request):
    fed_id = FederatedServer.fed_id
    FederatedServer.fed_id += 1
    return HttpResponse(fed_id, status.HTTP_200_OK)

@api_view(['GET'])
def reset(request):
    FederatedServer.reset()
    return HttpResponse("Request OK", status.HTTP_200_OK)
