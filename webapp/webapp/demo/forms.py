from email.policy import default
from django import forms
from .models import Downloader


class DownloaderForm(forms.ModelForm):
    consent = forms.BooleanField(required=True, label=(
        f"I consent to MedCAT collecting and storing my names, email, company"
        f" or academic institution name and use case description. I am aware"
        f" that MedCAT has been funded through academic research grants, and"
        f" therefore funding bodies require its support team to report wider"
        f" impact and usage of produced works using these information."
    ))
    class Meta:
        model = Downloader
        fields = [
            "first_name",
            "last_name",
            "email",
            "affiliation",
            "use_case",
        ]
        labels = {
            "first_name": "First Name",
            "last_name": "Last Name",
            "email": "Email",
            "affiliation": "Company or Academic Institution",
            "use_case": "Please describe your use case",
        }
        widgets = {
            "affiliation": forms.TextInput(attrs={"size": 40}),
            "use_case": forms.Textarea(attrs={"rows": 5, "cols": 40}),
        }
