# Cornell Conversational Analysis Toolkit ([ConvoKit](http://convokit.cornell.edu/))
This toolkit contains tools to extract conversational features and analyze social phenomena in conversations, using a [single unified interface](https://zissou.infosci.cornell.edu/socialkit/documentation/architecture.html) inspired by (and compatible with) scikit-learn.  Several large [conversational datasets](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit#datasets) are included together with scripts exemplifying the use of the toolkit on these datasets.

The toolkit currently implements features for:

### [Linguistic coordination](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html)

A measure of linguistic influence (and relative power) between individuals or groups based on their use of function words.  
Example: [exploring the balance of power in the US Supreme Court](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/coordination/examples.ipynb).

### [Politeness strategies](https://www.cs.cornell.edu/~cristian/Politeness.html)

A set of lexical and parse-based features correlating with politeness and impoliteness.  
Example: [understanding the (mis)use of politeness strategies in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations%20Gone%20Awry%20Prediction.ipynb).

### [Conversational prompts](http://www.cs.cornell.edu/~cristian/Asking_too_much.html)

An unsupervised method for extracting surface motifs that occur in conversations and grouping them by rhetorical role.  
Examples: [extracting common question types in UK parliament](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/question-typology/parliament_questions_example.ipynb), [understanding the use of conversational prompts in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations%20Gone%20Awry%20Prediction.ipynb).

### [Hypergraph conversation representation](http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html) 
A method for extracting structural features of conversations through a hypergraph representation.  
Example: [hypergraph creation and feature extraction, visualization and interpretation on a subsample of Reddit](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/hyperconvo/demo.ipynb).

## Datasets
These datasets are included for ready use with the toolkit:

### [Conversations Gone Awry Corpus](https://zissou.infosci.cornell.edu/socialkit/documentation/awry.html)

A collection of conversations from Wikipedia talk pages that derail into personal attacks (1,270 conversations, 6,963 comments)  
Name for download: `conversations-gone-awry-corpus`

### [Tennis Corpus](https://zissou.infosci.cornell.edu/socialkit/documentation/tennis.html)

Transcripts for tennis singles post-match press conferences for major tournaments between 2007 to 2015 (6,467 post-match press conferences).  
Name for download: `tennis-corpus`

### [Wikipedia Talk Pages Corpus](https://zissou.infosci.cornell.edu/socialkit/documentation/wiki.html)

A medium-size collection of conversations from Wikipedia editors' talk pages.  
Name for download: `wiki-corpus`

### [Supreme Court Corpus](https://zissou.infosci.cornell.edu/socialkit/documentation/supreme.html)

A collection of conversations from the U.S. Supreme Court Oral Arguments.  
Name for download: `supreme-corpus`

### [Parliament Corpus](https://zissou.infosci.cornell.edu/socialkit/documentation/parliament.html)

Parliamentary question periods from May 1979 to December 2016 (216,894 question-answer pairs).  
Name for download: `parliament-corpus`

### [Reddit Conversations Corpus](https://zissou.infosci.cornell.edu/socialkit/documentation/reddit-small.html) (beta)

99,145 Reddit conversations sampled from 100 subreddits.  
Name for download: `reddit-corpus`

These datasets can be downloaded using the `convokit.download()` [helper function](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/convokit/util.py).  Alternatively you can access them directly [here](http://zissou.infosci.cornell.edu/convokit/datasets/).


In addition to the provided datasets, you may also use ConvoKit with your own custom datasets by loading them into a `convokit.Corpus` object. [This example script](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/converting_movie_corpus.ipynb) shows how to construct a Corpus from custom data.

## Installation
This toolkit requires Python >= 3.6.

1. Download the toolkit: `pip3 install convokit`
2. Download Spacy's English model: `python3 -m spacy download en`

Alternatively, visit our [Github Page](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit) to install from source.

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/socialkit/documentation/). If you are new to ConvoKit, great places to get started are the [Core Concepts tutorial](https://zissou.infosci.cornell.edu/socialkit/documentation/architecture.html) for an overview of the ConvoKit "philosophy" and object model, and the [High-level tutorial](https://zissou.infosci.cornell.edu/socialkit/documentation/tutorial.html) for an walkthrough of how to import ConvoKit into your project, load a Corpus, and use ConvoKit functions.

## Acknowledgements

Andrew Wang ([azw7@cornell.edu](mailto:azw7@cornell.edu))  wrote the Coordination code and the respective example script, wrote the helper functions and designed the structure of the toolkit.

Ishaan Jhaveri ([iaj8@cornell.edu](mailto:iaj8@cornell.edu)) refactored the Question Typology code and wrote the respective example scripts.

Jonathan Chang ([jpc362@cornell.edu](mailto:jpc362@cornell.edu)) wrote the example script for Conversations Gone Awry.


[ConvoKit](http://convokit.cornell.edu/)
