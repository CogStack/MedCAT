
# DVC pipelines 

An interesting concept about DVC is the pipeline feature.

A brief explenation of what it does: it first creates and stores a setting inside the .dvc repo with the files required to run a pipeline (just the file names, not the actualy files), we start with the pipeline name by using the "-n" argument, then, we list the dependency files using the "-d" argument along with 1 file name, for multiple files you must specify the "-d" argument agian as in the example below. And since we will also have output files, we list them using the "-o" argument along with the file name, to ensure that they actually exist after the execution (note that these won't be added to the DVC/GIT repository by default, so if we wan't to actually stage these as part of the repository we have to manually add them using the dvc add command, don't forget to use git add after to add the .dvc files ), lastly, we just need to specify the actual script to run, in this case the bash script "run_pipeline.sh", it can also be a python file as long as we specify "python3 script.py".

## Setting the pipeline up

```
dvc run -n medcat_vocab_cdb_model_pipeline -d main.py -d pipeline_config.yaml -d models/vocab.dat -d models/cdb.dat -d data/cdb_input.txt -d data/vocab_input.txt -o models/vocab_out.dat -o models/cdb_out.dat -o models/annotation_unsupervised_out.dat /bin/bash run_pipeline.sh
```

Once we run the command, it will first attempt to check if all the dependencies are present, then if everything is valid, it runs the pipeline script specified at the end of the pipeline command.

If we need to modify a pipeline we can execute the same command again but with the "--force" (-f) parameter, this will overwrite the previous configuration for the pipeline with the declared name .
```
dvc run -n medcat_vocab_cdb_model_pipeline --force .....(rest of the params).....
```

## Reusing it

When we modify our data files or models (i.e the vocab, cdb file etc.) we only need to make sure that we keep the same file names as the ones decalred above in the dependencies. After we are done uploading/modifying new versions of our files all we need to do is rerun the pipleline using the "dvc repro" command. Dvc will then automatically see if any files in the dependency list has changed, and it will rerun the pipleline to generate the new files. After we are done we simply need to do a "dvc add" to the files we wish to track (if any), as the output files are not normally tracked, they will need to be updated manually.
