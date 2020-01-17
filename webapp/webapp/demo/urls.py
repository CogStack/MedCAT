from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('', train_annotations, name='train_annotations'),
]
