from django.db import models
from django.core.files.storage import FileSystemStorage


MODEL_FS = FileSystemStorage(location="/medcat_data")


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
    downloaded_file = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.first_name} - {self.last_name}'


class MedcatModel(models.Model):
    model_name = models.CharField(max_length=20, unique=True)
    model_file = models.FileField(storage=MODEL_FS)
    model_display_name = models.CharField(max_length=50)
    model_description = models.TextField(max_length=200)
