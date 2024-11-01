# CONSTANTs/ shouldn't change
REGRESSION_MODULE="medcat.utils.regression.regression_checker"
REGRESSION_OPTIONS="--strictness STRICTEST --require-fully-correct"

# CHANGABLES
# target models
DL_LINK="https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/all_fake_medcat_models.zip"
ZIP_FILE_NAME="all_fake_medcat_models.zip"
# target regression set
REGRESSION_TEST_SET="tests/resources/regression/testing/test_model_regresssion.yml"
# folder to house models under test
MODEL_FOLDER="fake_models"

# START WORK

echo "Downloading models"
wget $DL_LINK
# Create folder if it doesn't exit
mkdir -p "$MODEL_FOLDER"
echo "Uncompressing files"
unzip $ZIP_FILE_NAME -d $MODEL_FOLDER
echo "Cleaning up the overall zip"
rm $ZIP_FILE_NAME
for model_path in `ls $MODEL_FOLDER/*.zip`; do
    if [ -f "$model_path" ]; then
        echo "Processing $model_path"
        python -m $REGRESSION_MODULE \
            "$model_path" \
            $REGRESSION_TEST_SET \
            $REGRESSION_OPTIONS
        # this is a sanity check - needst to run after so that the folder has been created
        grep "MedCAT Version" "${model_path%.*}/model_card.json"
        # clean up here so we don't leave both the .zip'ed model
        # and the folder so we don't fill the disk
        echo "Cleaning up at: ${model_path%.*}"
        rm -rf ${model_path%.*}*
    else
        echo "No files found matching the pattern: $file"
    fi
done

# Remove the fake model folder
rm -r "$MODEL_FOLDER"
