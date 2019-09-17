from django.db import models

# Create your models here.
class UploadedText(models.Model):
    text = models.TextField(default="", blank=True)
    create_time = models.DateTimeField(auto_now_add=True)
