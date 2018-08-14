#### This is a module of [ConvoKit](http://convokit.cornell.edu/)

# Linguistic coordination
Linguistic coordination is a phenomenon in which people tend to (unconsciously) mimic the choices of function-word classes made by the people they are communicating with.  The degree of coordination has been shown to correlate with the relative status of the interlocutors (see the [Echoes of Power](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) paper).  

Linguistic coordination is related to [Linguistic Style Matching](http://journals.sagepub.com/doi/10.1177/026192702237953), [Lexical Entrainment](https://en.wikipedia.org/wiki/Lexical_entrainment) and the Communication Accommodation Theory (https://en.wikipedia.org/wiki/Communication_accommodation_theory).  See this [survey](https://www.annualreviews.org/doi/abs/10.1146/annurev-soc-081715-074206) for example applications of these concepts in computational social science.


## Example script
[Exploring the balance of power in the US Supreme Court and in the Wikipedia community of editors](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/coordination/examples.ipynb)

## Basic usage

We recommend using the example script above to familiarize yourself with this module of the toolkit, but here are basic steps:

0. Install [ConvoKit](http://convokit.cornell.edu/)
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create coordination object: `coord = convokit.Coordination(corpus)`
3. Define groups using `corpus.users`:
        `group_A = corpus.users(lambda user: user.info["is-justice"])`
4. Compute coordination: `scores = coord.score(group_A, group_B)`
5. (Optional) Get aggregate scores:
        `average_by_marker_agg1, average_by_marker, agg1, agg2, agg3 = coord.score_report(scores)`

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/socialkit/documentation/coordination.html).

#### [ConvoKit](http://convokit.cornell.edu/)

