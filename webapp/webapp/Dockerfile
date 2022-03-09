FROM python:3.7

# Create the required folders
RUN mkdir -p /webapp/models

# Copy everything
COPY . /webapp

ENV VOCAB_URL=https://medcat.rosalind.kcl.ac.uk/media/vocab.dat
ENV CDB_URL=https://medcat.rosalind.kcl.ac.uk/media/cdb-medmen-v1.dat

ENV CDB_PATH=/webapp/models/cdb.dat
ENV VOCAB_PATH=/webapp/models/vocab.dat

# Create the data directory
RUN mkdir -p /medcat_data

# Set the pythonpath
WORKDIR /webapp

RUN pip install -r requirements.txt

# Get the spacy model
RUN python -m spacy download en_core_web_md

# Build the db
RUN python manage.py makemigrations && \
    python manage.py makemigrations demo && \
    python manage.py migrate && \
    python manage.py migrate demo && \
    python manage.py collectstatic --noinput

# Create the db backup cron job
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils cron sqlite3 libsqlite3-dev
COPY etc/cron.d/db-backup-cron /etc/cron.d/db-backup-cron
RUN chmod 0644 /etc/cron.d/db-backup-cron
RUN crontab /etc/cron.d/db-backup-cron
