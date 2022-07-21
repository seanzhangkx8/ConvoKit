SPOLIN Corpus
============================

**Selected Pairs of Learnable ImprovisatioN (SPOLIN)** is a collection of more than 68,000 "Yes, and" type utterance pairs extracted from the long-form improvisation podcast *Spontaneanation* by Paul F. Tompkins, the Cornell Movie-Dialogs Corpus, and the SubTle corpus.


Dataset details
---------------

Speaker-level information
^^^^^^^^^^^^^^^^^^^^^^^^^
There is no speaker-level information.

Each conversation pair has two speakers named ``{conversation_id}_speaker_1`` for the first turn and ``{conversation_id}_speaker_2`` for the second turn.

Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every conversation is labeled with its source (Spontaneantion, Cornell Movie-Dialogs Corpus, or the SubTle corpus) and whether it abides by the “Yes, and” principle or not.
The "Yes, and" principle is a rule-of-thumb of improvisational theater that suggests that a participant should accept the reality of what the other participant has said (“Yes”) and expand or refine that reality with additional information ("and").
It does not require the response to explicitly contain the phrase "Yes, and".

Metadata for each utterance:

* split: whether it belongs to the original dataset’s train or validation set
* label: 1 if it is part of a "yes-and" pair or 0 otherwise
* source: whether it comes from Spontaneantion (``spont``), Cornell Movie-Dialogs Corpus (``cornell``), or the SubTle corpus (``subtle``)


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversation IDs are in the following format: ``{split}_{idx}``, where:

* split: either ``train`` or ``valid``, indicating whether the conversation belongs to the training set or the validation set.
* idx: an integer value that corresponds to the order that it appears in the original dataset.

Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

The metadata is as follows:

| {
|     "name": "spolin",
|     "brief description": "Selected Pairs of Learnable ImprovisatioN (SPOLIN) is a collection of more than 68,000 \"Yes, and\" type dialogue pairs extracted from the Spontaneanation podcast by Paul F. Tompkins, the Cornell Movie-Dialogs Corpus, and the SubTle corpus.",
|     "authors": "Hyundong Justin Cho, Jonathan May",
|     "poc_email": "jcho@isi.edu",
|     "github_url": "https://github.com/wise-east/spolin",
|     "publication_title": "Grounding Conversations with Improvised Dialogues",
|     "publication_venue": "ACL2020",
|     "publication_url": "https://aclanthology.org/2020.acl-main.218/",
|     "license": "Creative Commons Attribution-NonCommercial 4.0 International License",
| }

Usage
-----

To download directly with ConvoKit:

>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("spolin-corpus"))


For some quick stats:

>>> corpus.print_summary_stats()
Number of Speakers: 225194
Number of Utterances: 225194
Number of Conversations: 112597


**Number of yesands / non-yesands:**

* Spontaneanation: 10,959 / 6,087
* Cornell: 16,926 / 18,810
* SubTle: 40,303 / 19,512
* Total: 68,188 / 44,409

**Number of yesands / non-yesands (train split)**

* Spontaneanation: 10,459 / 5,587
* Cornell: 16,426 / 18,310
* SubTle: 40,303 / 19,512
* Total: 67,188 / 43,409

**Number of yesands / non-yesands (validation split)**

* Spontaneanation: 500 / 500
* Cornell: 500 / 500
* Total: 1,000 / 1,000

Additional notes
----------------

More details about the SPOLIN project can be found on: https://justin-cho.com/spolin

License
^^^^^^^
This dataset is shared under the `Creative Commons Attribution-NonCommercial 4.0 International License <https://creativecommons.org/licenses/by-nc/4.0/>`_.

Publication to cite
^^^^^^^^^^^^^^^^^^^

Please cite this paper when using it in your research:

| @inproceedings{cho2020spolin,
|     title={Grounding Conversations with Improvised Dialogues},
|     author={Cho, Hyundong and May, Jonathan},
|     booktitle ={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
|     year={2020}
| }


Contact
^^^^^^^

Please email any questions to Hyundong Justin Cho (jcho@isi.edu), Information Sciences Institute, University of Southern California
