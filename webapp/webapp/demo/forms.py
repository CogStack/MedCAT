from email.policy import default
from django import forms
from .models import Downloader


class DownloaderForm(forms.ModelForm):
    consent = forms.BooleanField(required=True, label=(
        f"I consent to MedCAT collecting and storing my names, email, company"
        f" or academic institution name, funder and project title, and use"
        f" case description. I am aware that MedCAT has been funded through"
        f" academic research grants, and therefore funding bodies require its"
        f" support team to report wider impact and usage of produced works"
        f" with the above information."
    ))

    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.fields["modelpack"] = forms.ChoiceField(label="Select a model for download",
                                                     choices=[(
                                                         model.model_name,
                                                         f"{model.model_display_name}{' (' + model.model_description + ')' if model.model_description else ''}"
                                                     ) for model in models],
                                                     widget=forms.RadioSelect())

    class Meta:
        model = Downloader
        exclude = ['downloaded_file']
        fields = [
            "first_name",
            "last_name",
            "email",
            "affiliation",
            "funder",
            "use_case",
        ]
        labels = {
            "first_name": "First Name",
            "last_name": "Last Name",
            "email": "Email",
            "affiliation": "Company or Academic Institution",
            "funder": "Funder and Project Title (optional)",
            "use_case": "Please describe your use case",
        }
        widgets = {
            "affiliation": forms.TextInput(attrs={"size": 40}),
            "funder": forms.TextInput(attrs={"size": 40}),
            "use_case": forms.Textarea(attrs={"rows": 5, "cols": 40}),
        }
