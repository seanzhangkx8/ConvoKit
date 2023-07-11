Choosing a Runtime Storage Backend
----------------------------------

The runtime storage backend determines how the data for ConvoKit objects is stored at runtime for use in Python code.
Note that this should not be confused with the data format, which defines how Corpus data gets saved to a file for persistence and redistribution.

ConvoKit supports two backends: native Python (the default) and MongoDB.
This guide provides a brief explanation of each one and why you might want to use them.

The Native Python Backend
=========================
ConvoKit's default backend stores all data in system memory as native Python objects (e.g., lists and dictionaries).
This means that once a Corpus has been loaded, reading and writing data to the Corpus is relatively fast, since all data is already in native Python representations so no conversion steps are needed.
However, this also involves keeping all the data in system memory (RAM) at once, which means that for large corpora the memory cost can be quite high.
Furthermore, since RAM is by definition volatile, any changes you make to the Corpus will persist only as long as your Python script or notebook stays running, unless you explicitly dump the Corpus to a file.
As a result, changes may be lost if your script crashes or your computer shuts down unexpectedly.

We generally recommend sticking to the default native Python backend due to its speed and simplicity.
In particular, it is a good choice for interactive sessions (e.g., ipython shell or Jupyter notebooks), where runtime errors won't crash the whole session/notebook and hence the persistence issue is less vital, and where you may be performing a lot of experimental changes to a Corpus that would benefit from the faster read/write speed.

The MongoDB Backend
===================
As the name suggests, the MongoDB backend stores all data in a MongoDB database.
This provides two key advantages.
First, it allows ConvoKit to take advantage of MongoDB's lazy loading, so not all data needs to be loaded into system memory at once, resulting in a much smaller memory footprint which enables the use of extremely large corpora in environments that might not otherwise have enough memory to handle them (e.g., a personal laptop).
Second, since all changes are written to the database, which is backed by on-disk files, changes are resilient to unexpected crashes; in the event of such a crash you can simply reconnect to the database to pick up where you left off.
On the other hand, reading and writing data to the Corpus is much slower in the MongoDB backend, both because doing so involves database reads/writes which are disk operations (slower than RAM operations) and because data must be converted between MongoDB format and Python objects.
Note that using the MongoDB backend requires some additional setup; see the :doc:`MongoDB setup guide </db_setup>` for instructions.

We recommend the MongoDB backend for the following use cases: memory-limited environments where your available RAM is insufficient to use your desired Corpus with the default backend; and live-service environments where you expect to continuously make changes to a Corpus over time and need to be resilient to unexpected crashes (e.g., using a Corpus as a component of a web server).

How to Change Backends
======================
Once you have chosen the backend that best suits your purposes, the next step is to tell ConvoKit to use it.
This can be done in three ways:

#. Corpus-level: ConvoKit supports specifying a backend on a per-Corpus basis. This is done through the ``storage_type`` parameter when constructing a corpus. You can set this parameter to the string ``"mem"`` for the native Python backend or ``"db"`` for the MongoDB backend. It is possible to mix Python-backed and MongoDB-backed corpora in the same script.

#. System-level: If you want to change the *default* backend in all ConvoKit code that runs on your computer (i.e., the backend that gets used when the ``storage_type`` parameter is not given), this is controlled by the ConvoKit system setting ``"default_storage_mode"``. This is set to ``"mem"`` when ConvoKit is first installed, but you can change it to ``"db"`` to tell ConvoKit to use the MongoDB backend by default. Note: ConvoKit system settings are found in the ``config.yml`` file, which is located in the hidden directory ``~/.convokit``.

#. Script-level: As an in-between option, if you want to change the default storage option used in a specific Python script but not at the whole-system level, you can do this by setting the environment variable ``CONVOKIT_STORAGE_MODE`` before running your script. For example, if you normally run your script as ``python3 myscript.py``, running it instead as ``CONVOKIT_STORAGE_MODE=db python myscript.py`` will set the default storage mode to MongoDB for that run of the script only.

Differences in Corpus behavior between backends
===============================================
For the most part, the two backends are designed to be interchangeable; that is, code written for one backend should work in the other backend out-of-the-box.
We made deliberate design choices during implementation to ensure consistent behavior of the code between the two supported backends.
However, some specifics of MongoDB result in one minor differences in Corpus behavior that you should be aware of when writing your code.

Since the MongoDB backend uses a MongoDB database as its data storage system, it needs to give that database a name.
Thus, there is an additional parameter in the Corpus constructor, db_collection_prefix, which is only used by the MongoDB backend.
This parameter determines how the MongoDB database will be named.
Note that you still have the option of not specifying a name, but in this case a random name will be used.
It is best practice to explicitly supply a name yourself, so you know what database to reconnect to in the event that reconnection is needed after a system crash.
