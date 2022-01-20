from django.db import models
from django.core.exceptions import ValidationError

# Create your models here.
class UploadedText(models.Model):
    text = models.TextField(default="", blank=True)
    create_time = models.DateTimeField(auto_now_add=True)


class Downloader(models.Model):
    first_name = models.CharField(max_length=20)
    last_name = models.CharField(max_length=20)
    email = models.EmailField(max_length=50)
    affiliation = models.CharField(max_length=100)
    funder = models.CharField(max_length=100, blank=True, default="")
    use_case = models.TextField(max_length=200)
