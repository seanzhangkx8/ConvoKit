Data Format
===========

ConvoKit expects and saves each corpus with the following basic structure, which mirrors closely with the intuitions behind the design of Corpus (see :doc:`architecture`).  

::

 corpus_directory
       |-- utterances.json
       |-- users.json
       |-- conversations.json
       |-- corpus.json
       |-- index.json

::

This corpus can be loaded with:

::

 corpus = Corpus(filename="corpus_directory")
::

Details of component files
--------------------------

utterances.json
^^^^^^^^^^^^^^^

Each utterance is represented as a json object, with six mandatory fields:

* id: index of the utterance
* user: the user who author the utterance
* root: index of the conversation root of the utterance
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Additional information can be added optionally, depending on characteristics of the dataset and intended use, as:

* meta: dictionary of utterance metadata

utterances.json contains a list of such utterances. An example utterance is shown below, drawn from the Supreme Court corpus: 

::

 {'id': '200', 'user': 'mr. srinivasan', 'root': '145', 'reply_to': '199', 'timestamp': None, 'text': 'It -- it does.', 'meta': {'case': '02-1472', 'side': 'respondent'}}
::


users.json
^^^^^^^^^^

Users are identified by user names. users.json keeps a dictionary, where the keys are user names, and values are metadata associated with the users. Provision of user metadata is optional.  

An example user-metadata pair is shown below, again, drawn from the Supreme Court corpus: 

::

'mr. srinivasan': {'is-justice': False, 'gender': 'male'}

::


conversation.json 
^^^^^^^^^^^^^^^^^

Similarly, conversation.json also keeps a dictionary where keys are conversation index, and values are conversational-level metadata (i.e., additional information that stay invariant throughout the conversation). 

An example conversation index-metadata pair is shown below, adapted from the conversations gone awry corpus: 

::

"236755381.13326.13326": {"page_title": "User talk: Entropy", "conversation_has_personal_attack": true}

::

Provision of conversational-level metadata is optional. In case no information is provided, the file could simply contain an empty dictionary.  


corpus.json
^^^^^^^^^^^

Metadata of the corpus is saved in corpus.json, as a dictionary where keys are names of the metadata, and values are the actual content of such metadata. 

The contents of the corpus.json file for the Reddit corpus (small) is as follows: 

::

 {"subreddit": "reddit-corpus-small", "num_posts": 8286, "num_comments": 288846, "num_user": 119889}

::


index.json 
^^^^^^^^^^

To allow users the option of previewing available information in the corpus without loading it entirely, ConvoKit requires an index.json file that contains information about all available metadata and their expected types. 

There are five mandatory fields: 

* utterances-index: information of utterance-level metadata
* users-index: information of user-level metadata
* conversations-index: information of conversation-level metadata
* overall-index: information of corpus-level metadata
* version: version number of the corpus

As an example, the corpus-level metadata for the Reddit corpus (small) is shown below: 

::

"overall-index": {"subreddit": "<class 'str'>", "num_posts": "<class 'int'>", "num_comments": "<class 'int'>", "num_users": "<class 'int'>"}
:: 

 
Converting from custom datasets
-------------------------------

To convert a custom dataset into the ConvoKit data format, one may directly transform the custom dataset into the set of component files according to the specifications explained above. 

Alternatively, one can also rely on ConvoKit to handle most of conversion, by transforming the source data into a list of Utterance objects, and create a corresponding corpus object with:

>>> corpus = Corpus(utterances = custom_utterance_list) 

The dataset can be saved with ConvoKit format for reuse by: 

>>> corpus.dump("custom_dataset")

A detailed example of how the `Cornell Movie--Dialogs Corpus <https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html>`_. can be converted from its original release form to ConvoKit format can be found `here <>`_.  
