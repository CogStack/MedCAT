#!/usr/bin/env bash

SSHD_CONFIG_PATH="/etc/ssh/sshd_config"

echo "Adding settings"

isInFile=$(cat $SSHD_CONFIG_PATH | grep -c "serverPermitLocalCommand yes")

if [ $isInFile -eq 0 ]; then
   echo "serverPermitLocalCommand yes" >> $SSHD_CONFIG_PATH
fi

export HOME="/model_repository/"