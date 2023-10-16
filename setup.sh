#!/bin/bash

cpu=0
local=0
# passes extra commands from CLI for pip
pip_extras=""

# Checks if CPU torch is required, or if local medcat install is required
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c | --cpu) cpu=1 ;;
        -l | --local) local=1 ;;
        *) pip_extras="${pip_extras} ${1}" ;;
    esac
    shift
done

command="python3 -m pip install"

if (($local==1));
then
command="${command} ."
else
command="${command} medcat"
fi

if (($cpu==1)); 
then
    command="${command} --extra-index-url https://download.pytorch.org/whl/cpu/";
fi

command="${command}${pip_extras}"

eval "$command"