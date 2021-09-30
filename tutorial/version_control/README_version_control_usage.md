# General Usage

First we need to configure the app to use our git credentials for actions as well as authentication, to do so, execute: 

```python3 -m medcat config```

Then go through the steps, please ignore the "storage_server_*" variables, they are only used for testing purposes at the moment.

If configured properly, we should be able to see the models available to use by doing:

```python3 -m medcat listmodels```

If this doesn't work then please run the `configure` command again and make sure all the settings are correct, and that you have access to the GitHub repo (if it's the case, and it is not a public repo).

To being downloading a model please execute :

```python3 -m medcat download MODEL_NAME```

    - the cli should begin the download process
    - after it downloads the repository files it will most likely ask for a new set of credentials that are used to get the actual model files from the storage server.

Finally, when you are done with your training, you should package your model release: 

```python3 -m medcat package```
