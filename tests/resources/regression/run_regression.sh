#!/bin/bash

# exit immediately upon non-zero exit status
set -e

# create and train model and capture output
output=$(python tests/resources/regression/creation/cat_creation.py)
# make sure the user sees the output
echo "$output"

# extract the last line of the output which contains the full model path
model_path=$(echo "$output" | tail -n 1)
# NOTE: this file should be tagged with the python version we're using

# run the regression_checker with the captured file path
python -m medcat.utils.regression.regression_checker \
  "$model_path" \
  tests/resources/regression/testing/test_model_regresssion.yml \
  --strictness STRICTEST \
  --require-fully-correct

# Step 4: Clean up the generated file
rm -f "$model_path"
