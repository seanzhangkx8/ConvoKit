#### This is a module of [ConvoKit](http://convokit.cornell.edu/)

# Hypergraph Conversation Model
Tools to build hypergraph models of conversations and extract features from them (see the [Patterns of Participant Interactions](http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html) paper).

## Example script
[Understanding the (mis)use of politeness strategies in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/hyperconvo/example-visualization.ipynb)

## Basic usage

We recommend using the example script above to familiarize yourself with this module of the toolkit, but here are basic steps:

0. Install [ConvoKit](http://convokit.cornell.edu/)
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create HyperConvo object: `hc = convokit.HyperConvo(corpus)`
3. Extract hypergraph features: `feats = hc.retrieve_feats()`
4. Optionally, use convenience methods to embed threads or communities in a low-dimensional space for further exploration:

        X_threads, roots = hc.embed_threads(feats)
        X_communities, subreddits = hc.embed_communities(feats, "subreddit")

#### [ConvoKit](http://convokit.cornell.edu/)
