#### This is a module of [ConvoKit](http://convokit.cornell.edu/)

# Question Typology and Conversational Prompts
This module implements a method for grouping questions and other comments according to their the rhetorical intentions  (see the [Asking too much](http://www.cs.cornell.edu/~cristian/Asking_too_much.html) and [Conversations gone awry](http://www.cs.cornell.edu/~cristian/Conversations_gone_awry.html) papers).  The types of rhetorical intentions are extracted automatically and in an unsupervised fashion from a given collection of conversations.

## Example scripts

Extracting common question types in the [UK parliament](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/questionTypology/parliament_question_typology.py), on [Wikipedia edit pages](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/questionTypology/wiki_question_typology.py), and in [sport interviews](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/questionTypology/tennis_question_typology.py).

[Understanding the use of conversational prompts in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversationsGoneAwry/Conversations%20Gone%20Awry%20Prediction.ipynb)


## Basic usage

We recommend using the example scripts above to familiarize yourself with this module of the toolkit, but here are basic steps:

0. Install [ConvoKit](http://convokit.cornell.edu/)
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create QuestionTypology object (discover typology): `questionTypology = QuestionTypology(`
3. Explore 10 questions of type `type_num`: `questionTypology.display_questions_for_type(type_num, 10)`
4. Explore 10 resulting motifs of type `type_num`: `questionTypology.display_motifs_for_type(type_num, 10)`
5. Explore 10 resulting answer fragments from answers to questions of type `type_num`: `questionTypology.display_answer_fragments_for_type(type_num, 10)`
6. Explore 10 question-answer pairs from the training data of type `type_num`: `questionTypology.display_question_answer_pairs_for_type(type_num, 10)`

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/socialkit/documentation/questionTypology.html).


#### [ConvoKit](http://convokit.cornell.edu/)
