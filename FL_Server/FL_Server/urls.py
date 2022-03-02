from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('index', views.index),
    path('admin/', admin.site.urls),
    path('initialize/<int:client_num>/<int:experiment>/<int:max_round>', views.initialize),
    path('get_server_round', views.get_server_round),
    path('get_compile_config', views.get_compile_config),
    path('get_server_model', views.get_server_model),
    path('get_server_weight', views.get_server_weight),
    path('put_local_weight/<int:client_id>', views.put_local_weight),
    path("update_num_data/<int:client_id>/<int:num_data>", views.update_num_data),
    path('reset/', views.reset),
 ]
