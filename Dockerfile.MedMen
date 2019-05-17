FROM python:3.7

COPY . /cat
# Set the pythonpath
WORKDIR /cat

RUN pip install medcat
# Reqs for the api
RUN pip install -r requirements.txt

# Get the spacy model
#RUN python -m spacy download en_core_web_sm
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_md-0.2.0.tar.gz

# Get the MedMen models
RUN wget "https://s3-eu-west-1.amazonaws.com/zkcl/med_ann_norm.dat" -P /cat/models/
RUN wget "https://s3-eu-west-1.amazonaws.com/zkcl/med_ann_norm_dict.dat" -P /cat/models/

# Now run the simple api
WORKDIR /cat/api
ENTRYPOINT ["python"]
CMD ["api.py"]
