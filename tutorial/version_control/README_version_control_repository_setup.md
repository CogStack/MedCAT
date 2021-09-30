# How DVC works

The idea behind dvc is to have one standard git repository containing only the references to the data & model files used for machine learning. This git repo will also contain a basic python file for running a pipeline that generates new model files for use (eventually scalable  so that it pulls data from multiple locations).

# Run the docker container 

docker-compose -f vc_repo_docker_compose.yml up -d

Copy your add .pub key file to the `docker_ssh_config` folder or add the key to `authorized_keys`.

# Installing DVC and other repo requirements

Dvc can be installed in multiple ways, but luckily, there is a python package maintend by it's developers, and is included in the requirements.txt file.

```
python3 -m pip install -r requirements.txt
```

or, if we only want to install dvc:

```
python3 -m pip install dvc
```

# STEP 1 : Creating a repository 

Start by initialising a GIT repository first.

``` 
mkdir cogstack_medcat_dvc_repo
cd cogstack_medcat_dvc_repo
git init 
```

Add the git remote location, since we intend to store the repository on another server.

``` 
git remote add remote_name https://github.com/user/reponame.git
```

or, particularly useful for institutional and private deployments ( NOT SUPPORTED AT THE MOMENT):

```
git remote add origin master ssh://repoaddress/path/to/repo/folder

```
PLEASE note that the above repository setup won't work currently as the implementation only supports GITHUB Repos.


To view the remote list, execute ```git remote -v``` .

Initialize a DVC repo (REQUIRED), this step is essential and must be done before doing anything else within the repository.

``` 
dvc init 
```

At this stage a dvc repo is initialized, and we are ready to start adding files to it in a similar way to GIT.


# STEP 1A (optional) : Adding files and commiting changes  

Tthis should only be done for non-model files that are necessary ONLY ( for example, custom MODELCARDS)

We must first add the folder containing the models and the data files (if any) so that they can be tracked by DVC

We will do this using the dvc add path/file command

For example, to add a folder with files inside. Pay attention to the command, we specify the files within a folder, not a folder name. This is because if we just do "dvc add data_folder", dvc will assume that wee want to track the folder (withou the data in it only, so it doesnt create links for the files inside)

```
dvc add data_folder/*  
```

or for individual files

```
dvc add example_folder/filename1.ext example_folder/filename2.ext etc.
```

Example model card addition : 

```
dvc add modelcard.md
```


Why are we doing it this way and not simply by just using git add folder ? Simple, dvc doesnt really support folder additions, because it is made to track large files, we would usually manually add them or use a pattern e.g model_*.dat .

Once a file has been added, DVC will create a .gitignore file that tells git which files to ignore from commiting. We need this because we don't want to be cloning the whole dataset if we clone this git repo, we only need the references to the large files to be committed. 
DVC will also create .dvc files, which we are going to commit to the repository using the git add command, these files will always have the same name as the files we added, with the ".dvc" extension name added to them.

```
git add example_folder/*.dvc
```

If we receive a message about adding files from a folder which is not tracked normally, we cau use the -f parameter to force the addition(s), usually not necessary, use this with caution.

```
git add -f example_folder/*.dvc
```

We should now also commit the stage of the data files we just added to DVC by doing :
```
dvc commit
```

# STEP 2 : Adding remote(s)

This needs to be done so that users can pull data from other sources. Essentially we just configure a storage location for files that are tracked by DVC only.

### Local system folder or other sources
#
Supports most storage locations:

Google Drive:

``` 
	dvc remote add --default myremote gs://my-bucket/path
```

You will have to make sure that you have gs-utils installed and configured systemwide with the keys from the google-storage api in order for this to work.

SSH:
``` 
	dvc remote add --default myremote ssh://user@example.com/path/to/dir
```

### To add SSH server as remote(s) with preconfigured credentials ###
#
If you wish to add a default server to which dvc will always attempt to pull/push things:
``` 
dvc remote add --default ssh-storage-name ssh://example.com/path/to/storage
```
Example using KCL Rosalind as storage:

```
dvc remote add ssh-storage-name ssh://username@login.rosalind.kcl.ac.uk:/home/username/
```
or non-username bound
```
dvc remote add -d ssh-storage-name ssh://rosalind.kcl.ac.uk/path/to/folder 
```
Scratch groups, will require configuration as explained below: 

```
dvc remote add -d ssh-storage-name ssh://login.rosalind.kcl.ac.uk/scratch/groups/scratch_group_name/repo_folder/
```

 To store the configurations localy only, use the --local option, all configurations will be stored in the repository folder .dvc/config.local


``` 
dvc remote modify --local ssh-storage-name ask_password true
dvc remote modify --local ssh-storage-name port 22
```

With this setting, you should have a ssh key on the server to allow connections if ask_password is false, otherwise this may not work
 (it will say that the password or public key's dont match)

Once this is done, the basic storage repo configuration is set up, we are now ready to push the state of the repo to the master branch.

 To store the connection data only for your own user or for the entire system

User: use the --global parameter
Entire system: if you are an administrator and wish to store the DVC remote data across the system use the --system option

More info: https://dvc.org/doc/command-reference/config

### Local folder example ( stored models inside a local folder)

dvc remote add -d ssh-storage-name "/path/to/my/own/folder"

Please include drive letters / FULL absolute filesystem path to that folder, otherwise there might be issues.

An example dvc config file is available `sample_dvc_config` in the current folder. It will store all models in the /models/tutorial/version_control/ folder by default.
Useful for running the unit tests with `python3 -m unittest medcat/tests/cli/version_control/package_tests.py`.

# STEP 3 : Commit the changes to the base repo.

``` 
dvc commit
```

Afterwards , we can safely commit the git files.

``` 
git commit -m "Commit message"
```


#### If we have global remotes that are configured on the current system and wish to import them we can use the dvc import-url to pull from a custom remote

dvc import-url remote://ssh-storage-name/path/to/file


### Initialize DVC hooks

Since we don't really want to repeat the dvc add steps over and over again every time we change a file (a data file that is) we should attempt to make the process easier so that everytime we do modify a file and we want to create a new commit, it is done automatically when we execute git commit.


```
dvc install
``` 

And attempt to push the files to our remote server.
IMPORTANT: Git hooks are not pushable on the remote server, you will have to manually execute this command everytime you pull the repository in a new location.

```
dvc push
```

We can now also push the changes to the git repository.

```
git push -u remote_name master
```

And now we should have a fully functioning repo, ready to be used with the medcat version control cli feature !


## Why did we have to do the above steps ?

Pretty much necessary as we need to track and stage/commit all the files. We use git only to track the placeholders (.dvc) of the datafiles. DVC stores the other files on a separate storage bare repository. 


# STEP 4 (OPTIONAL) Manually Updating/staging files that are already in the DVC repo. (Should not be used normally)

If we have a file that has been updated, say for example our model file in "models/vocab.dat", we need to first stage this change in dvc by using the dvc add command again :

```
dvc add "models/vocab.dat"
```

This will update the dvc file for the vocab.dat file with a newly generate hash.

We will need to commit this update to the git repo as well now.

```
git add models/vocab.dat.dvc
```

# N/A in the current version : SETTING UP CUSTOM REPOSITORY (non-git environments) 


To setup a bare repository on the main server (which will be handling the data repository) 

```
mkdir cogstack_dvc_repo
cd cogstack_dvc_repo
git --bare init

```

If we want to get the data from the server, we need to do a ```dvc pull``` first.
We should then have all the files we need !

The provided hook script should be used in the bare repository (that does not have the raw files of the git repo), like the one initialized eariler.
