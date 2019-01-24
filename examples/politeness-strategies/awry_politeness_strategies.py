# This example extracts politeness strategies from the Conversations Gone Awry dataset,
#   one of the steps in the Conversations Gone Awry paper (http://www.cs.cornell.edu/~cristian/Conversations_gone_awry.html).
#   For code reproducing the full results of the paper, see the example notebook in the
#   `conversations-gone-awry` example subdirectory.

from convokit import PolitenessStrategies, Parser, Corpus, download

corpus = Corpus(filename=download('conversations-gone-awry-corpus'))

# the PolitenessStrategies module requires spacy parses
parser = Parser()
corpus = parser.transform(corpus)

# extract the politeness strategies.
# Note: politeness strategies are a hand-engineered feature set, so no fitting is needed.
ps = PolitenessStrategies(verbose=100)
corpus = ps.transform(corpus)

print("Showing politeness strategies for 10 example utterances")
for utt_id in corpus.get_utterance_ids()[:10]:
    print("Utterance", utt_id)
    print("Content:", corpus.get_utterance(utt_id).text)
    print("Detected politeness strategies:")
    strats = corpus.get_utterance(utt_id).meta["politeness_strategies"]
    for strat_name in strats:
        if strats[strat_name] == 1:
            print("\t%s" % strat_name)