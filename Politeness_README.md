#### This is a module of [ConvoKit](http://convokit.cornell.edu/)

# Politeness strategies
Politeness strategies are a set of lexical and syntactic features that correlate with politeness and impoliteness (see the [A computational approach to politeness](https://www.cs.cornell.edu/~cristian/Politeness.html) paper).  This module extracts such strategies from conversational data.

Check out [politness.cornell.edu](http://politeness.cornell.edu/) for more politeness-related resources.

## Example script
[Understanding the (mis)use of politeness strategies in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations%20Gone%20Awry%20Prediction.ipynb)

## Basic usage

We recommend using the example script above to familiarize yourself with this module of the toolkit, but here are basic steps:

0. Install [ConvoKit](http://convokit.cornell.edu/)
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create PolitenessStrategies object: `ps = convokit.PolitenessStrategies(corpus)`
3. Explore the binary politeness strategies features for a single comment from the corpus: `ps[comment_id]`
4. Alternatively, get the features for the whole corpus in the form of a Pandas DataFrame: `ps.feature_df`

#### [ConvoKit](http://convokit.cornell.edu/)
