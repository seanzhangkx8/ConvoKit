import socialkit

# set up corpus 
corpus = socialkit.Corpus(filename=socialkit.download("supreme-corpus"))
coord = socialkit.Coordination(corpus)

# get set of all justices
justices = corpus.users(lambda user: user.info["is-justice"])
# get set of all users
everyone = corpus.users()

# compute coordination from each justice to everyone
print("Justices, ranked by how much they coordinate to others:")
justices_to_everyone = coord.score(justices, everyone)
for justice, score in sorted(justices_to_everyone.averages_by_user().items(),
        reverse=True):
    print(justice.name, round(score, 5))
print()

# compute coordination from everyone to each justice
print("Justices, ranked by how much others coordinate to them:")
coord_to_justices = {}
for justice in justices:
    everyone_to_justice = coord.score(everyone, [justice])
    coord_to_justices[justice] = coord.score_report(everyone_to_justice)
for justice, scores in sorted(coord_to_justices.items(),
    key=lambda item: item[1][4], reverse=True):  # sort by aggregate 3 score 
    print(justice.name, round(scores[4], 5))
print()
