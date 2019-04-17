Reddit corpus (small)
=====================
 
A representative sample of conversations from Reddit from 100 highly active subreddits (each subreddit has at least 100 comments threads that has at least 10 comments each). The complete list of subreddits included can be found `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/subreddits.txt>`_. 

Dataset details
---------------

User-level information
^^^^^^^^^^^^^^^^^^^^^^

Users in this corpus are reddit users, identified by their account name. We include their activity statistics within the corpus:

* num_posts: number of post this user has authored
* num_comments: number of comment activities from the user


Utterance-level information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each individual post and comment is viewed as an utterance. For each utterance, we include:

* score: score (i.e., the number of upvotes minus the number of downvotes) of the content 
* top_level_comment: the id of the top level comment
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
* domain: [?]
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
>>> len(corpus.get_conversations())
8285

Additional note
---------------
