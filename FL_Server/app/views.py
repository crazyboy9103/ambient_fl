import json
from django.http import HttpResponse
# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import JSONParser
from app import numpy_encoder
from app.Federated import FederatedServer
import json

@api_view(['GET'])
def index(request):
    return HttpResponse("index ok", status.HTTP_200_OK)

@api_view(['GET'])
def get_server_round(request):
    try:
        return HttpResponse(FederatedServer.get_server_round(), status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to get server round", status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_compile_config(request):
    try:
        compile_config = json.dumps(FederatedServer.get_compile_config()) # optim, loss
        return HttpResponse(compile_config, status.HTTP_200_OK)

    except:
        return HttpResponse("Failed to get compile config", status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_experiment(request):
    try:
        json_data = json.dumps(FederatedServer.experiment)
        return HttpResponse(json_data, status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to get experiment number", status.HTTP_500_INTERNAL_SERVER_ERROR)
       
@api_view(['PUT'])
def set_experiment(request, experiment):
    if experiment not in [1,2,3,4]:
        return HttpResponse("Set experiment to one of 1,2,3,4", status.HTTP_400_BAD_REQUEST)
    try:
        FederatedServer.experiment = experiment
        return HttpResponse(f"Experiment set to {FederatedServer.experiment}", status.HTTP_200_OK)

    except:
        return HttpResponse("Failed to set experiment", status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_server_model(request):
    try:
        return HttpResponse(FederatedServer.get_model_as_json(), status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to fetch the keras model", status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])
def set_client_num(request, client_num):
    try:
        FederatedServer.client_number = client_num
        return HttpResponse(f"Total client number is set to {client_num}", status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to fetch client number", status.HTTP_200_OK)

@api_view(['GET'])
def get_server_weight(request):
    try:
        server_weight = FederatedServer.get_server_weight()
        server_weight_to_json = json.dumps(server_weight, cls=numpy_encoder.NumpyEncoder)
        return HttpResponse(server_weight_to_json, status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to fetch server weight", status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['PUT'])
def put_weight(request, client_id):
    try:
        weight = JSONParser().parse(request)
        return HttpResponse(FederatedServer.update(client_id, weight), status.HTTP_200_OK)
    except:
        weight = None
        return HttpResponse(FederatedServer.update(client_id, weight), status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def update_num_data(request, client_id, num_data):
    try:
        return HttpResponse(FederatedServer.update_num_data(client_id, num_data), status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to update data num", status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def reset(request):
    try:
        FederatedServer.reset()
        return HttpResponse("Server reset", status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to reset server", status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def initialize(request, client_num, experiment, max_round):
    try:
        return HttpResponse(FederatedServer.initialize(client_num, experiment, max_round), status.HTTP_200_OK)
    except:
        return HttpResponse("Failed to initialize server", status.HTTP_500_INTERNAL_SERVER_ERROR)
