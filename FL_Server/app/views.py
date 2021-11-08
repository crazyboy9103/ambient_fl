import json
from django.http import HttpResponse
# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from app import numpy_encoder
from app.Federated import FederatedServer

@api_view(['GET'])
def index(request):
    logger.info("request index")
    return HttpResponse("index ok", status.HTTP_200_OK)

@api_view(['GET'])
def get_round(request):
    return HttpResponse(FederatedServer.get_current_round(), status.HTTP_200_OK)
@api_view(['GET'])
def get_server_round(request):
    return HttpResponse(FederatedServer.get_server_round(), status.HTTP_200_OK)
@api_view(['GET'])
def get_experiment(request):
    json_data = json.dumps(FederatedServer.experiment)
    return HttpResponse(json_data, status.HTTP_200_OK)
       
@api_view(['PUT'])
def set_experiment(request, experiment):
    if experiment in [1,2,3,4]:
        FederatedServer.experiment = experiment
    else:
        FederatedServer.experiment = 1
    return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['PUT'])
def put_accuracy(request, client_id):
    json_data = JSONParser().parse(request)
    FederatedServer.accuracy[client_id] = float(json_data['accuracy'])
    return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['PUT'])
def set_client_num(request, client_num):
    FederatedServer.client_number = client_num
    return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['GET'])
def get_server_weight(request):
    server_weight = FederatedServer.get_server_weight()
    server_weight_to_json = json.dumps(server_weight, cls=numpy_encoder.NumpyEncoder)
    return HttpResponse(server_weight_to_json, status.HTTP_200_OK)

@api_view(['PUT'])
def put_weight(request, client_id):
    json_data = JSONParser().parse(request)
    weights = list(json_data['weights'])
    FederatedServer.update(client_id, weights)
    return HttpResponse("Request PUT OK", status.HTTP_200_OK)

@api_view(['PUT'])
def update_num_data(request, client_id, num_data):
    return HttpRespose(FederatedServer.update_num_data(client_id, num_data), status.HTTP_200_OK)

@api_view(['GET'])
def get_num_data(request):
    return HttpResponse(FederatedServer.total_num_data, status.HTTP_200_OK)

@api_view(['GET'])
def reset(request):
    FederatedServer.reset()
    return HttpResponse("Request OK", status.HTTP_200_OK)

@api_view(['GET'])
def initialize(request, client_num, experiment, max_round):
   
    return HttpResponse(FederatedServer.initialize(client_num, experiment, max_round), status.HTTP_200_OK)

