Reddit corpus (small)
=====================
 
A representative sample of conversations from Reddit from 100 highly active subreddits (each subreddit has at least 100 comments threads that has at least 10 comments each during September, 2018). The complete list of subreddits included can be found `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/subreddits.txt>`_. 

Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this corpus are Reddit users, identified by their account names. The corpus includes the following activity statistics:

* num_posts: number of posts from the user
* num_comments: number of comments from the user


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each individual post or comment is viewed as an utterance. For each utterance, the corpus include:

* score: score (i.e., the number of upvotes minus the number of downvotes) of the content 
* top_level_comment: the id of the top level comment (None if the utterance is a post)
* retrieved_on: unix timestamp of the time of when the data is retrieved 
* gilded: gilded status of the content
* gildings: gilding information of the content
* stickied: stickied status of the content
* permalink: permanent link of the content
* author_flair_text: flair of the author 


Conversational-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each post with its corresponding comments are considered a conversation. For each conversation, we provide:

* title: title of the post
* num_comments: number of comments following this post
* domain: domain of the post
* subreddit: subreddit this post is retrieved from
* gilded: gilded status of the post
* gildings: gilding information of the post
* stickied: stickied status of the post
* author_flair_text: flair of the author 


Corpus-level information
^^^^^^^^^^^^^^^^^^^^^^^^

* subreddit: the list of subreddits included in this corpus 
* num_posts: total number of posts included in this corpus
* num_comments: total number of comments in this corpus
* num_user: number of unique users in this corpus


Usage
-----

To download directly with ConvoKit: 

>>> corpus = Corpus(filename=download("reddit-corpus-small"))

For some quick stats:

>>> len(corpus.get_utterance_ids()) 
282418
>>> len(corpus.get_usernames())
115431
>>> len(corpus.get_conversation_ids())
8285

Additional note
---------------

Refer to :doc:`tutorial` for a quick tutorial with ConvoKit with this corpus. 
