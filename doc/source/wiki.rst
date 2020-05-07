Wikipedia Talk Pages Corpus
===========================

A collection of conversations from `Wikipedia editor's talk pages <http://en.wikipedia.org/wiki/Wikipedia:Talk_page_guidelines>`_. with metadata. 

Distributed together with: `Echoes of power: Language effects and power differences in social interaction <https://www.cs.cornell.edu/~cristian/Echoes_of_power.html>`_. Cristian Danescu-Niculescu-Mizil, Lillian Lee, Bo Pang, and Jon Kleinberg. WWW 2012.

Dataset details
---------------

speaker-level information
^^^^^^^^^^^^^^^^^^^^^^

speakers in this dataset are Wikipedia editors; their account names are taken as the speaker names. Additional information include:

* is-admin: whether the speaker is an admin
* gender: gender of the speaker
* edit-count: total number of edits the speaker has made


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each utterance, we provide:

* id: index of the utterance
* speaker: the speaker who author the utterance
* root: index of the conversation root of the utterance
* reply_to: index of the utterance to which this utterance replies to (None if the utterance is not a reply)
* timestamp: time of the utterance
* text: textual content of the utterance

Metadata for each utterance include:

* is-admin: whether the utterance is from an admin
* parsed: parsed version of the utterance text, represented as a SpaCy Doc


Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("wiki-corpus"))


For some quick stats:

>>> len(corpus.get_utterance_ids()) 
391294
>>> len(corpus.get_speakernames())
38462
>>> len(corpus.get_conversation_ids())
125292


A common use case for the WikiConv corpora might be to combine corpora from different years for further analysis. This is straightforward with the Corpus's merge functionality, which we demonstrate below.

>>> from convokit import Corpus, download
>>> wikiconv_2003 = Corpus(filename=download("wikiconv-2003"))
>>> wikiconv_2003.print_summary_stats()
Number of Speakers: 9168
Number of Utterances: 140265
Number of Conversations: 91787
>>> wikiconv_2004 = Corpus(filename=download("wikiconv-2004"))
>>> wikiconv_2004.print_summary_stats()
Number of Speakers: 34235
Number of Utterances: 613608
Number of Conversations: 303494
>>> merged_corpus = wikiconv_2003.merge(wikiconv_2004)
WARNING: Multiple values found for Speaker([('name', 'TUF-KAT')]) for meta key: speaker_id. Taking the latest one found
>>> merged_corpus.print_summary_stats()
Number of Speakers: 41509
Number of Utterances: 753873
Number of Conversations: 392883

Notice that the number of Utterances in the merged corpus is simply the sum of those in the constituent corpora. This is to be expected since the Utterances from these two corpora are from different years and are therefore distinct and non-overlapping.

However, the number of speakers and Conversations is not the sum of those of the constituent corpora -- undoubtedly because some speakers have made Utterances in both years and because some Conversations took place over 2003 and 2004.


Additional notes
----------------

Contact
^^^^^^^

Please email any questions to: cristian@cs.cornell.edu (Cristian Danescu-Niculescu-Mizil)







