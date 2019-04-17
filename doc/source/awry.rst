Conversations gone awry corpus
==============================

A collection of conversations from Wikipedia talk pages that derail into personal attacks (1,270 conversations, 6,963 comments). 

Distributed together with: `Conversations gone awry: Detecting early signs of conversational failure <https://www.cs.cornell.edu/~cristian/Conversations_gone_awry_files/conversations_gone_awry.pdf>`_. Justine Zhang, Jonathan P. Chang, Cristian Danescu-Niculescu-Mizil, Lucas Dixon, Yiqing Hua, Nithum Thain, Dario Taraborelli. ACL 2018. 


Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this dataset are Wikipedia editors; their account names are taken as the user names. 

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata for each utterance include:

* is_section_header: whether the utterance is a conversation "title" or "subject" as seen on the original talk page (if true, this utterance should be ignored when doing any NLP tasks)
* comment_has_personal_attack: whether this comment was judged by 3 crowdsourced workers to contain a personal attack
* parsed: parsed version of the utterance text, represented as a SpaCy Doc


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[How are conversations arranged?] Metadata for each conversation include:

* page_title: the title of the talk page the comment came from
* page_id: the unique numerical ID of the talk page the comment came from
* pair_id: the conversation ID (root) of the conversation that this comment's conversation is paired with
* conversation_has_personal_attack: whether any comment in this comment's conversation contains a personal attack
* verified: whether the personal attach label has been verified to be correct 
* pair_verified: whether the personal attach label for the paired conversation has been verified to be correct


Usage
-----

To download directly with ConvoKit: 

>>> corpus = Corpus(filename=download("conversations-gone-awry-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
6960
>>> len(corpus.get_usernames())
2146
>>> len(corpus.get_conversations())
1270


Additional note
---------------

This data was collected from late 2017 to early 2018 and was annotated in April 2018.


Related links
^^^^^^^^^^^^^
`Wikipedia editors' talk Pages <http://en.wikipedia.org/wiki/Wikipedia:Talk_page_guidelines>`_.


Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)







