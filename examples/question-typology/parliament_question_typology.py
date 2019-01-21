# This example extracts question types from the UK Parliament Question Answer Sessions
#   reproducing the asking too much paper (http://www.cs.cornell.edu/~cristian/Asking_too_much.html).
#   (due to the non-deterministic nature of clustering, the order of the clusters and some cluster assignments
#    will vary from the clusters published in the paper, but since there is a seed provided, multiple executions
#    of this script will always produce the same clusters)

from convokit import Corpus, Parser, QuestionTypology, download

print("Loading Parliament dataset...")
corpus = Corpus(filename=download("parliament-corpus"))

# Get parses for each utterance in the Corpus. This step is needed since the
# QuestionTypology will use the parses in its computation.
parser = Parser()
corpus = parser.fit_transform(corpus)

# initialize the QuestionTypology Transformer. Note the following differences 
# from the original parliament_question_typology.py:
#  - We do not pass in a corpus, as the Transformer-based API will
#    instead be having us apply the QuestionTypology to a corpus
#    after initialization
#  - We do not pass in a dataset name, for the same reason as above
#  - We do not pass in a data_dir, as serialization is now manual
questionTypology = QuestionTypology(num_dims=25, num_clusters=8, verbose=10000, random_seed=164)

print("Fitting QuestionTypology...")
corpus = questionTypology.fit_transform(corpus)

questionTypology.display_totals()
print("10 examples for types 1-8:")
for i in range(8):
    questionTypology.display_motifs_for_type(i, num_egs=10)
    questionTypology.display_answer_fragments_for_type(i, num_egs=10)
    questionTypology.display_question_answer_pairs_for_type(i, num_egs=10)

print("Example cluster assignments and distances for utterances in the corpus:")
n_printed = 0
for utt_id in corpus.get_utterance_ids():
    if n_printed == 10:
        break
    utterance = corpus.get_utterance(utt_id)
    if "qtype" in utterance.meta:
        print("Utterance %s: %s" % (utt_id, utterance.text))
        print("Cluster assignment:", utterance.meta["qtype"])
        print("Cluster distances:", utterance.meta["qtype_dists"])
        n_printed += 1
