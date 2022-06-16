from django.contrib import admin
from .models import *

admin.site.register(Downloader)
admin.site.register(MedcatModel)

def remove_text(modeladmin, request, queryset):
    UploadedText.objects.all().delete()

class UploadedTextAdmin(admin.ModelAdmin):
    model = UploadedText
    actions = [remove_text]

# Register your models here.
admin.site.register(UploadedText, UploadedTextAdmin)

