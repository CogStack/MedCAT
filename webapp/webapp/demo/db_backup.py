import os
from django.core import management
from django.conf import settings
from django_cron import CronJobBase, Schedule


class DbBackup(CronJobBase):

    RUN_EVERY_MINS = int(os.environ.get("DB_BACKUP_EVERY_MINS", 60 * 12))
    RETRY_AFTER_FAILURE_MINS = int(os.environ.get("DB_BACKUP_RETRY_BACKOFF_MINS", 5))

    schedule = Schedule(run_every_mins=RUN_EVERY_MINS, retry_after_failure_mins=RETRY_AFTER_FAILURE_MINS)
    code = "demo.db_backup.DbBackup"

    def __init__(self):
        backup_location = settings.DBBACKUP_STORAGE_OPTIONS["location"]
        os.makedirs(backup_location, exist_ok=True)

    def do(self):
        management.call_command("dbbackup", "--noinput", "-z")
