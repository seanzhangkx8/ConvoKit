Setting Up MongoDB For ConvoKit
===============================

`The MongoDB Documentation <https://docs.mongodb.com/>`_ provides a complete 
guide on installing and running a MongoDB server. Here, we provide a simplified 
guide to getting MongoDB setup to use with ConvoKit's DB Storage mode, in a handful
of settings. 

Running MongoDB with Conda
--------------------------

0. Install conda if needed, following `these instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_ for your system.
1. (Optional) Create a new environment where you want to install mongodb:

:: 

 $ conda create --name my_env

2. Activate your newly created environment, or an existing environment where you want to install mongodb:

:: 

 $ conda activate my_env


3. Install the mongodb package.

:: 

 $ conda install mongodb

Check to see if version is at least 5.0. 

::

 $ mongod --version

If not, utilize:

::
 
 $ conda install -c conda-forge mongodb=5.0


4. Start the MongoDB server as a daemon process.

:: 

 $ mongod --fork --logpath <path for DB logs> --dbpath <path for DB storage>

5. Use the MongoDB server for ConvoKit!
6. To stop the MongoDB server, on Linux or MacOS, use the ``htop`` command to find the mongod process ID and run:

:: 

 $ kill <mongod process ID>

6. Alternitivly, to stop the MongoDB server on Linux, run

:: 

 $ mongod --shutdown  


Sometimes, the above process doesn't work for MacOS. However, there is another solution for MacOS users below.


Running MongoDB on MacOS with Homebrew
--------------------------------------

0. If needed install Homebrew `here <https://brew.sh/>`_.
1. Use Homebrew to install MongoDB.

::

 $ brew tap mongodb/brew
 $ brew install mongodb-community@5.0

2. Start MongoDB.

::

 $ brew services start mongodb-community@5.0

3. Use the MongoDB server for ConvoKit!
4. To stop the MongoDB server, run

:: 

 $ brew services stop mongodb-community@5.0

Using MongoDB Atlas: A remote MongoDB server in the cloud
---------------------------------------------------------

MongoDB offers a cloud service version of their database, called MongoDB Atlas.
Atlas provides a free tier that is a good option for starting out with ConvoKit 
remote DB storage, and several paid tiers that provide production level performance. 
Follow these instructions, based on `the instructions for getting started with Atlas
provided by the MongoDB team <https://docs.atlas.mongodb.com/getting-started/>`_, 
to setup a MongoDB server in the cloud for use with ConvoKit.

0. Register a new MongoDB Atlas account here: https://account.mongodb.com/account/register, and log into the Atlas UI.
1. Create a new MongoDB cluster and a database user within the Atlas UI.
2. Add your IP address to the set of approved IP addresses that can connect to cluster, and setup a DB user, within the Atlas UI (as suggested in the "Setup connection security" tab).
3. In the "Choose a connection method" tab, select "Connect your Application" and choose Python as your driver. Then, copy the outputted URI, which should look something like ``mongodb+srv://<dbusername>:<dbpassword>@cluster0.m0srt.mongodb.net/myFirstDatabase?retryWrites=true&w=majority``
4. Paste the aforementioned URI into ~/.convokit/config.yml in the db_host field. Then, replace <dbusername> and <dbpassword> with the credentials you setup in step 1, and replace ``myFirstDatabase`` with ``convokit``. 
5. Use the remote MongoDB server for ConvoKit!