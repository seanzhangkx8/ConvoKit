Wikipedia Talk Pages Corpus
===========================

A collection of conversations from `Wikipedia editor's talk pages <http://en.wikipedia.org/wiki/Wikipedia:Talk_page_guidelines>`_. with metadata. 

Distributed together with: `Echoes of power: Language effects and power differences in social interaction <https://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_. Cristian Danescu-Niculescu-Mizil, Lillian Lee, Bo Pang, and Jon Kleinberg. WWW 2012.

Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this dataset are Wikipedia editors; their account names are taken as the user names. Additional information include:

* is-admin: whether the user is an admin
* gender: gender of the user
* edit-count: total number of edits the user has made 


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata for each utterance include:

* is-admin: whether the utterance is from an admin
* parsed: parsed version of the utterance text, represented as a SpaCy Doc


Usage
-----

To download directly with ConvoKit: 

>>> corpus = Corpus(filename=download("wiki-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
391294
>>> len(corpus.get_usernames())
38462
>>> len(corpus.get_conversation_ids())
125292


Additional note
---------------

Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)







