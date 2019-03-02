# This example uses the wikipedia conversations corpus to reproduce figure 1
#   from the echoes of power paper (https://www.cs.cornell.edu/~cristian/Echoes_of_power.html).
#
# The plots answer these questions: 
# - Do users on the whole coordinate more to admins or nonadmins?
# - Do admins coordinate to other people more than nonadmins do?

from convokit import Utterance, Corpus, Coordination, download

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# load corpus; split users by whether they are an admin
# this means that if a user has spoken in the corpus as both an admin and
#   a non-admin, then we will split this user into two users, one for each of
#   these roles
corpus = Corpus(filename=download("wiki-corpus"))
split = ["is_admin"]

# create coordination object
coord = Coordination()
coord.fit(corpus)

# helper function to plot two coordination scores against each other as a chart,
#   on aggregate and by coordination marker
# a is a tuple (speakers, targets)
# b is a tuple (speakers, targets)
def make_chart(a_scores, b_scores, a_description, b_description, a_color="b", b_color="g"):
    # get scores by marker and on aggregate
    _, a_score_by_marker, a_agg1, a_agg2, a_agg3 = coord.score_report(corpus, a_scores)
    _, b_score_by_marker, b_agg1, b_agg2, b_agg3 = coord.score_report(corpus, b_scores)

    # the rest plots this data as a double bar graph
    a_data_points = sorted(a_score_by_marker.items())
    b_data_points = sorted(b_score_by_marker.items())
    a_data_points, b_data_points = zip(*sorted(zip(a_data_points, b_data_points),
        key=lambda x: x[0][1], reverse=True))
    labels, a_data_points = zip(*a_data_points)
    _, b_data_points = zip(*b_data_points)

    labels = ["aggregate 1", "aggregate 2", "aggregate 3"] + list(labels)
    a_data_points = [a_agg1, a_agg2, a_agg3] + list(a_data_points)
    b_data_points = [b_agg1, b_agg2, b_agg3] + list(b_data_points)

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(a_data_points)) + 0.35)
    ax.set_xticklabels(labels, rotation="vertical")

    ax.bar(np.arange(len(a_data_points)), a_data_points, 0.35, color=a_color)
    ax.bar(np.arange(len(b_data_points)) + 0.35, b_data_points, 0.35, color=b_color)

    a_scores_a1 = [s for s in a_scores if len(a_scores[s]) == 8]
    b_scores_a1 = [s for s in b_scores
            if len(b_scores[s]) == 8]
    b_patch = mpatches.Patch(color="b",
                             label=a_description + " (total: " +
                             str(len(a_scores_a1)) + ", " +
                             str(len(a_scores)) + ")")
    g_patch = mpatches.Patch(color="g",
                             label=b_description + " (total: "  +
                             str(len(b_scores_a1)) + ", " +
                             str(len(b_scores)) + ")")
    plt.legend(handles=[b_patch, g_patch])

    filename = str(a_description) + " vs " + str(b_description) + ".png"
    plt.savefig(filename, bbox_inches="tight")
    print('Created chart "' + filename + '"')

# get all groups of users that we want to compare
everyone = corpus.iter_users()
admins = corpus.iter_users(lambda u: u.meta["is-admin"])
nonadmins = everyone - admins

# do users on the whole coordinate more to admins or nonadmins?
make_chart(
    coord.score(corpus, everyone, admins, focus="targets", target_thresh=7,
        split_by_attribs=split),
    coord.score(corpus, everyone, nonadmins, focus="targets", target_thresh=7,
        split_by_attribs=split),
    "Target: admins", "Target: nonadmins"
)
# do admins coordinate to other people more than nonadmins do?
make_chart(
    coord.score(corpus, admins, everyone, speaker_thresh=7, target_thresh=7,
        split_by_attribs=split),
    coord.score(corpus, nonadmins, everyone, speaker_thresh=7, target_thresh=7,
        split_by_attribs=split),
    "Speaker: admins", "Speaker: nonadmins"
)
