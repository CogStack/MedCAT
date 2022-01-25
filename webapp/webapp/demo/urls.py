from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('', train_annotations, name='train_annotations'),
    path('auth_callback', validate_umls_user, name='validate_umls_user'),
    path('download_model', download_model, name="download_model")
]
