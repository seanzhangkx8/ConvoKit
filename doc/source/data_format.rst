Data Formats
============

ConvoKit expects and saves each corpus as a set of json files, each covering relevant information for one class in the class hierarchy explained in ([todo] add link to Corpus section in architecture). 

Utterance
^^^^^^^^^

Information about the main contents of the conversations (i.e., the utterances) is represented as a list of 

User
^^^^
Participants involved in a Corpus is represented as 

Conversation
^^^^^^^^^^^^

Corpus
^^^^^^

Index 
^^^^^
 
Converting from custom datasets
-------------------------------

To convert a custom dataset into the ConvoKit data format, one may transform the source data into a list of Utterance objects, and create a corresponding corpus object with:
:: 
 corpus = Corpus(utterances = utterance_list) 
::

The dataset can be saved with ConvoKit format for reuse by: 
:: 
 corpus.dump(“example_dataset”)
:: 

A detailed example of how the `Cornell Movie--Dialogs Corpus <https://www.cs.cornell.edu/~cristian/Chameleons_in_imagined_conversations.html>`_. may be converted from its original release form to ConvoKit format can be found [todo: add link] here. 

Alternatively, one could also directly transform a dataset into the set of json files according the the specifications above. 
