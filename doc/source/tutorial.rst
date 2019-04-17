====================
Quick-start tutorial
====================

Setup
=====
Read the :ref:`introduction to Convokit <README>` and the description of its :ref:`architecture`.

This toolkit requires Python 3.

If you haven't already,

#. Download the toolkit: ``pip3 install convokit``

#. Download Spacy's English model: ``python3 -m spacy download en``

Interactive tutorial
====================
Let us start an interactive session (e.g. with ``python`` or ``ipython``) and import Convokit.

>>> import convokit

Now we load an existing corpus, specifically: `reddit-corpus-small`.

By design, it includes 100 comment threads (each consisting of at least 10 Utterances) from 100 popular subreddits from September 2018.

>>> corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

Alternatively, if you have a custom corpus, i.e. a corpus directory (say, "corpus-dir") containing the corpus component files:

* conversations.json

* corpus.json

* index.json

* users.json

* utterances.json

We can construct a corpus from that directory.

>>> filepath = "corpus-dir" # update the filepath accordingly
>>> corpus = convokit.Corpus(filename=filepath)

Exploring the corpus
--------------------

We can examine the corpus metadata:

>>> corpus.meta
{'num_comments': 288846,
 'num_posts': 8286,
 'num_user': 119889,
 'subreddit': 'reddit-corpus-small'}

So the corpus includes 288,846 comments and 8286 posts. This is a total of 297,132 Utterances. (An Utterance is either a post or a comment in a reddit corpus.)

These 297,132 Utterances were made by 119889 different users.

We can get iterators of Utterances, Users, and Conversations, and confirm their sizes match the metadata.

>>> len(list(corpus.iter_users()))
119889
>>> len(list(corpus.iter_utterances()))
297132
>>> len(list(corpus.iter_conversations()))
8286

We can also get a list of Utterance ids.

>>> utter_ids = corpus.get_utterance_ids()

Let's confirm that there are 297132 Utterances as expected.

>>> len(utter_ids)
297132

Let's take the first Utterance id and examine the Utterance it corresponds to:

>>> utter_ids[0]
'9c716m'
>>> corpus.get_utterance(utter_ids[0])
Utterance({'id': '9c716m', 'user': User([('name', 'AutoModerator')]), 'root': '9c716m', 'reply_to': None, 'timestamp': 1535839576, 'text': 'Talk about your day. Anything goes, but subreddit rules still apply. Please be polite to each other! \n', 'meta': {'score': 13, 'top_level_comment': None, 'retrieved_on': 1540061887, 'gilded': 0, 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0}, 'subreddit': 'singapore', 'stickied': False, 'permalink': '/r/singapore/comments/9c716m/rsingapore_random_discussion_and_small_questions/', 'author_flair_text': ''}})

Let's explore the Utterance object further.

>>> utt = corpus.get_utterance(utter_ids[0])
>>> utt.meta # Utterance-level metadata
{'score': 13,
 'top_level_comment': None,
 'retrieved_on': 1540061887,
 'gilded': 0,
 'gildings': {'gid_1': 0, 'gid_2': 0, 'gid_3': 0},
 'subreddit': 'singapore',
 'stickied': False,
 'permalink': '/r/singapore/comments/9c716m/rsingapore_random_discussion_and_small_questions/',
 'author_flair_text': ''}
>>> utt.timestamp # the unix timestamp for when the utterance was posted
1535839576
>>> utt.user # the User who posted the Utterance
User([('name', 'AutoModerator')])
>>> utt.user.meta # User-level metadata
{'num_posts': 200, 'num_comments': 27}

Applying a transformer
----------------------

We initialize a HyperConvo transformer, which extracts structural features of conversations through a hypergraph representation.

>>> # Limit hypergraph representation to threads of length at least 10, using the first 10 utterances
>>> # include_root is set to False as we only want comment threads (i.e. threads that begin with the top level comment, not the original post.)
>>> hc = convokit.HyperConvo(prefix_len=10, min_thread_len=10, include_root=False)
>>> hc.fit_transform(corpus)
>>> corpus.meta.keys()
dict_keys(['subreddit', 'num_posts', 'num_comments', 'num_user', 'hyperconvo'])
>>> corpus.meta # warning: outputs a lot of text
{'subreddit': 'reddit-corpus-small',
 'num_posts': 8286,
 'num_comments': 288846,
 'num_user': 119889,
 'hyperconvo': {'e58slx0': {'max[outdegree over c->c responses]': 1,
   'max[indegree over c->c responses]': 3,
   'argmax[outdegree over c->c responses]': 1,
   'argmax[indegree over c->c responses]': 1,
   'norm.max[outdegree over c->c responses]': 0.1111111111111111,
   'norm.max[indegree over c->c responses]': 0.3333333333333333,
   '2nd-largest[outdegree over c->c responses]': 1,
   '2nd-largest[indegree over c->c responses]': 3,
   '2nd-argmax[outdegree over c->c responses]': 2,
......

The output of the HyperConvo transformer is stored in the Corpus metadata.






