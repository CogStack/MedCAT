FROM python:3.7

# Create the required folders
RUN mkdir -p /webapp/models

# Copy everything
COPY . /webapp

ENV VOCAB_URL=https://s3-eu-west-1.amazonaws.com/zkcl/vocab.dat
ENV CDB_URL=https://s3-eu-west-1.amazonaws.com/zkcl/cdb-medmen.dat

ENV CDB_PATH=/webapp/models/cdb.dat
ENV VOCAB_PATH=/webapp/models/vocab.dat

# Set the pythonpath
WORKDIR /webapp

RUN pip install -r requirements.txt

# Get the spacy and scipspacy model
RUN python -m spacy download en_core_web_md
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz

# Build the db
RUN python manage.py makemigrations && \
    python manage.py makemigrations demo && \
    python manage.py migrate && \
    python manage.py migrate demo 
