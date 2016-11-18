import socialkit

# set up model 
model = socialkit.Model(filename="../../datasets/supreme-corpus/full.json")
coord = socialkit.Coordination(model)

# get set of all justices
justices = model.user_names(lambda user: user.info["is-justice"])
# get set of all users
everyone = model.user_names()

# compute coordination from each justice to everyone
print("Justices, ranked by how much they coordinate to others:")
justices_to_everyone = coord.score(justices, everyone)
for justice, scores in sorted(justices_to_everyone.items(),
        key=lambda item: sum(item[1].values()), reverse=True):
    print(justice, round(sum(scores) / len(scores), 5))
print()

# compute coordination from everyone to each justice
print("Justices, ranked by how much others coordinate to them:")
coord_to_justices = {}
for justice in justices:
    everyone_to_justice = coord.score(everyone, justice)
    coord_to_justices[justice] = coord.score_report(everyone_to_justice)
for justice, scores in sorted(coord_to_justices.items(),
        key=lambda item: item[1][4], reverse=True):  # sort by aggregate 3 score 
    print(justice, round(scores[4], 5))
print()
