# Politeness strategies
Politeness strategies are a set of lexical and parse-based features that correlate with politeness and impoliteness (see the [A computational approach to politeness](https://www.cs.cornell.edu/~cristian/Politeness.html) paper). This readme contains information about  the [Installation](#installation) and [Basic Usage](#basic-usage).

## Installation
1. The toolkit requires Python 3. If you don't have it install it by running `pip install python3` or using the Anaconda distribution. That can be found [here](https://www.anaconda.com/download/#macos).
2. Install the required packages by running `pip install -r requirements.txt` (Note if your default version of `pip` is for Python 2.7 you might have to use `pip3 install -r requirements.txt` instead)
3. Run `python3 setup.py install` to install the package.
4.  Use `import convokit` to import it into your project.

## Basic usage
1. Load corpus: `corpus = convokit.Corpus(filename=...)`
2. Create PolitenessStrategies object: `ps = convokit.PolitenessStrategies(corpus)`
3. Explore the binary politeness strategies features for a single comment from the corpus: `ps[comment_id]`
4. Alternatively, get the features for the whole corpus in the form of a Pandas DataFrame: `ps.feature_df`

