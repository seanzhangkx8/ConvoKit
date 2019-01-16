import spacy
import sys

from .transformer import Transformer

class Parser(Transformer):
    """
    Transformer that adds SpaCy parses to each Utterance in a Corpus. This 
    parsing step is a prerequisite for most of the models included in convokit.

    :param spacy_nlp: if provided, the Parser will use this SpaCy object to do
        parsing. Otherwise, it will initialize a SpaCy object via load('en')
    """

    def __init__(self, spacy_nlp=None):
        self.ATTR_NAME = "parsed"
        if spacy_nlp is not None:
            self.spacy_nlp = spacy_nlp
        else:
            try:
                self.spacy_nlp = spacy.load('en')
            except OSError:
                print("Convokit requires a SpaCy English model to be installed. Run `python -m spacy download en` and retry.")
                sys.exit()

    def transform(self, corpus):
        """Runs the SpaCy parser on each utterance in the corpus, and adds the 
        parses to the utterance metadata table.

        :return: corpus, modified with parses assigned to each utterance
        """
        for ut in corpus.iter_utterances():
            parsed = self.spacy_nlp(ut.text)
            ut.meta[self.ATTR_NAME] = parsed
        return corpus