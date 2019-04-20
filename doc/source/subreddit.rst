Subreddit corpus
================

A collection of Corpuses of Reddit data built from `Pushshift.io Reddit Corpus <https://pushshift.io/>`_. Each Corpus contains posts and comments from an individual subreddit from its inception until 2018-10. 

A total of 948,169 subreddits are included, the list of subreddits included in the dataset can be explored `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/>`_. Note that the directories are ordered lexicographically, with capital letters sorted first, so the subreddit KDC is found in JustinYCult~-~Kanye/ rather than Kanye2024~-~Kemetic/.

We also provide a small sample of the collection (see :doc:`reddit-small`): Reddit corpus (small). 

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

A subreddit corpus name is always the name of the subreddit with the prefix "subreddit-". For example, the subreddit `Cornell <https://www.reddit.com/r/Cornell>`_. can be downloaded as follows: 

>>> corpus = Corpus(filename=download("subreddit-Cornell"))

For some quick stats on this subreddit:

>>> len(corpus.get_utterance_ids()) 
74467 
>>> len(corpus.get_usernames())
7568
>>> len(corpus.get_conversation_ids())
10744


Additional note
---------------

1. Some subreddit corpus is big. If the subreddit of interest is highly active, it is advised to check the size of the compressed subreddit corpus file `here <https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/>`_. before downloading.

2. This is a beta version release. Not all subreddits that exist are included, and the completeness of subreddit history is not guaranteed. In some cases, the user activity information (i.e., number of posts/comments) may be inflated by duplicated entries in intermediate processing steps. We anticipate further updates to fix existing issues and to provide a more complete version of the dataset. 

