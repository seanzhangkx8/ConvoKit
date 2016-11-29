from convokit import Utterance, Corpus, Coordination, download

from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pickle

if os.path.isfile("wiki-coord.p"):
    coord = pickle.load(open("wiki-coord.p", "rb"))
    print("Loaded corpus from wiki-coord.p")
else:
    corpus = Corpus(filename=download("wiki-corpus"))
    print("Loaded corpus")
    coord = Coordination(corpus)
    coord.precompute()
    print("Done precomputing")
    pickle.dump(coord, open("wiki-coord.p", "wb"))

corpus = coord.corpus
corpus.subdivide_users_by_attribs(["is-admin"])
everyone = corpus.users()
admins = corpus.users(lambda u: u.info["is-admin"])
nonadmins = everyone - admins

# each of a and b should be a tuple (speakers, targets)
def make_chart(a, b, a_label, b_label):
    s1, t1 = a
    s2, t2 = b
    admin_scores = coord.score(s1, t1, utterances_thresh_indiv=4,
            target_thresh=1)
    admin_a1m, admin_m, admin_agg1, admin_agg2, admin_agg3 = \
            coord.score_report(admin_scores)
    nonadmin_scores = coord.score(s2, t2, utterances_thresh_indiv=4,
            target_thresh=1)
    nonadmin_a1m, nonadmin_m, nonadmin_agg1, nonadmin_agg2, nonadmin_agg3 = \
            coord.score_report(nonadmin_scores)

    admins = sorted(admin_m.items())
    nonadmins = sorted(nonadmin_m.items())
    admins, nonadmins = zip(*sorted(zip(admins, nonadmins),
        key=lambda x: x[0][1], reverse=True))
    labels, admins = zip(*admins)
    _, nonadmins = zip(*nonadmins)

    labels = ["aggregate 1", "aggregate 2", "aggregate 3"] + list(labels)
    admins = [admin_agg1, admin_agg2, admin_agg3] + list(admins)
    nonadmins = [nonadmin_agg1, nonadmin_agg2, nonadmin_agg3] + list(nonadmins)

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(admins)) + 0.35)
    ax.set_xticklabels(labels, rotation="vertical")

    ax.bar(np.arange(len(admins)), admins, 0.35, color="b")
    ax.bar(np.arange(len(nonadmins)) + 0.35, nonadmins, 0.35, color="g")

    admin_scores_a1 = [s for s in admin_scores if len(admin_scores[s]) == 8]
    nonadmin_scores_a1 = [s for s in nonadmin_scores
            if len(nonadmin_scores[s]) == 8]
    b_patch = mpatches.Patch(color="b",
                             label=a_label + " (total: " +
                             str(len(admin_scores_a1)) + ", " +
                             str(len(admin_scores)) + ")")
    g_patch = mpatches.Patch(color="g",
                             label=b_label + " (total: "  +
                             str(len(nonadmin_scores_a1)) + ", " +
                             str(len(nonadmin_scores)) + ")")
    plt.legend(handles=[b_patch, g_patch])
    plt.show()

make_chart((everyone, admins), (everyone, nonadmins),
        "Target: admins", "Target: nonadmins")
#make_chart((admins, everyone), (nonadmins, everyone),
        #"Speaker: admins", "Speaker: nonadmins")
