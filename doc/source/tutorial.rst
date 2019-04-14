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

Now we load an existing corpus, specifically: `reddit-corpus-small`.

By design, it includes 100 comment threads (each consisting of at least 10 utterances) from 100 popular subreddits from October 2018.

>>> corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

Alternatively, if you have a custom corpus, i.e. a corpus directory (say, "corpus-dir") containing the corpus component files:

* conversations.json

* corpus.json

* index.json

* users.json

* utterances.json

We can construct a corpus from that directory.

>>> filepath = "corpus-dir"
>>> corpus = convokit.Corpus(filename=filepath)

Exploring the corpus
--------------------

We can examine the corpus metadata:

>>> corpus.meta
{'num_comments': 3606,
 'num_posts': 156,
 'num_user': 1449,
 'subreddit': 'reddit-corpus-small'}

So the corpus includes 3606 comments and 156 posts. This is a total of 3762 Utterances. (An Utterance is either a post or a comment in a reddit corpus.)

These 3762 Utterances were made by 1449 different users.

We can get iterators of Utterances, Users, and Conversations, and confirm their sizes match the metadata.

>>> len(list(corpus.iter_users()))
1449
>>> len(list(corpus.iter_utterances()))
3762
>>> len(list(corpus.iter_conversations()))
156

We can also get a list of Utterance ids.

>>> utter_ids = corpus.get_utterance_ids()

Let's confirm that there are 3762 Utterances as expected.

>>> len(utter_ids)
3762

Let's take the first Utterance id and examine the Utterance it corresponds to:

>>> utter_ids[0]
'9c0kpy'
>>> corpus.get_utterance(utter_ids[0])
Utterance({'id': '9c0kpy', 'user': User([('name', 'LamPard31')]), 'root': '9c0kpy', 'reply_to': None, 'timestamp': 1535778431, 'text': '', 'meta': {'score': 780, 'top_level_comment': None, 'retrieved_on': 1540058138, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'singapore', 'stickied': False, 'permalink': '/r/singapore/comments/9c0kpy/first_world_chinese/', 'author_flair_text': ''}})

Let's explore the Utterance object further. An Utterance contains its own metadata.

>>> utt = corpus.get_utterance(utter_ids[0])
>>> utt.meta
{'author_flair_text': '',
 'gilded': 0,
 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0},
 'permalink': '/r/singapore/comments/9c18wm/seeing_the_other_chinese_compo_i_thought_id_post/',
 'retrieved_on': 1540058593,
 'score': 146,
 'stickied': False,
 'subreddit': 'singapore',
>>> utt.timestamp # the unix timestamp for when the utterance was posted
1535786508
>>> utt.user # the User who posted the Utterance
User([('name', 'ThatCalisthenicsDude')])
>>> utt.user.meta # User-level metadata
{'num_comments': 2, 'num_posts': 1}

Applying a transformer
----------------------








