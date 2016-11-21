from socialkit import Utterance, Model, Coordination

from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pickle

# cache results in a pickle
if os.path.isfile("supreme-coord.p"):
    coord = pickle.load(open("supreme-coord.p", "rb"))
else:
    model = Model(filename="../../datasets/supreme-corpus/full.json")
    model.subdivide_users_by_attribs(["case", "justice-is-favorable"])
    coord = Coordination(model)
    coord.precompute()
    pickle.dump(coord, open("supreme-coord.p", "wb"))

model = coord.model
everyone = model.users()
justices = model.users(lambda u: u.info["is-justice"])
lawyers = model.users(lambda u: not u.info["is-justice"])
fav_justices = model.users(lambda u: u.info["is-justice"] and
        u.info["justice-is-favorable"])
unfav_justices = model.users(lambda u: u.info["is-justice"] and
        not u.info["justice-is-favorable"])
print(len(everyone))
x = input()

# each of a and b should be a tuple (speakers, targets)
def make_chart(a, b, a_label, b_label, a_color="b", b_color="g"):
    s1, t1 = a
    s2, t2 = b
    admin_scores = coord.score(s1, t1, target_thresh=5)
    admin_a1m, admin_m, admin_agg1, admin_agg2, admin_agg3 = \
            coord.score_report(admin_scores)
    nonadmin_scores = coord.score(s2, t2, target_thresh=5)
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

    ax.bar(np.arange(len(admins)), admins, 0.35, color=a_color)
    ax.bar(np.arange(len(nonadmins)) + 0.35, nonadmins, 0.35, color=b_color)

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

make_chart((justices, lawyers), (lawyers, justices),
        "Justices to lawyers", "Lawyers to justices", "g", "b")
make_chart((lawyers, unfav_justices), (lawyers, fav_justices),
        "Target: unfavorable justice", "Target: favorable justice")
make_chart((unfav_justices, lawyers), (fav_justices, lawyers),
        "Speaker: unfavorable justice", "Speaker: favorable justice")

#make_chart((admins, everyone), (nonadmins, everyone))
#fig, ax = plt.subplots()
#rects1 = ax.bar(np.arange(len(target_admins)), target_admins,
#                0.35, color="b")
#rects2 = ax.bar(np.arange(len(target_nonadmins)) + 0.35, target_nonadmins,
#                0.35, color="g")
#ax.set_xticks(np.arange(len(target_nonadmins)) + 0.35)
#ax.set_xticklabels(labels, rotation='vertical')
#b_patch = mpatches.Patch(color='blue',
#                         label='Target: Admins (total: ' +
#                         str(n_target_admins[0]) + ", " +
#                         str(n_target_admins[2]) + ")")
#g_patch = mpatches.Patch(color='green',
#                         label='Target: Non-admins (total: '  +
#                         str(n_target_nonadmins[0]) + ", " +
#                         str(n_target_nonadmins[2]) + ")")
#plt.legend(handles=[b_patch, g_patch])
#plt.title("Targets")


