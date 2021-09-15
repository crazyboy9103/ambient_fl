from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('', views.index),
    path('index', views.index),
    path('admin/', admin.site.urls),
    path('round/', views.round),
    path('round', views.round),
    path('client_num', views.client_num),
    path('client_num/', views.client_num),
    path('weight', views.weight),
    path('weight/', views.weight),
    path('reset/', views.reset),
    path('get_id', views.get_id),
    path('total_num', views.total_num_data),
    path('total_num/', views.total_num_data),
    path('experiment', views.experiment),
    path('experiment/', views.experiment),
    path('accuracy', views.accuracy),
    path('accuracy/', views.accuracy)
 ]
