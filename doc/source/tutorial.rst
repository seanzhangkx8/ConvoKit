====================
Quick-start tutorial
====================

Setup
=====
Read the :ref:`introduction to Convokit <README>` and the description of its :ref:`Architecture`.

This toolkit requires Python 3.5.

If you haven't already,

#. Download the toolkit: ``pip3 install convokit``

#. Download Spacy's English model: ``python3 -m spacy download en``

Interactive tutorial
====================
Let us start an interactive session (e.g. with ``python`` or ``ipython``) and import Convokit.

>>> import convokit

Now we load an existing corpus, specifically: ``reddit-corpus-small``.

By design, it includes 100 comment threads (each consisting of at least 10 utterances) from 100 popular subreddits from October 2018.

>>> corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

Exploring the corpus
--------------------

Let's verify that it contains what we expect.




