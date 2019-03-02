# This example uses the supreme court corpus to compute some simple information:
# - Which justices coordinate the most to others?
# - Which justices are coordinated to the most?

import convokit

# set up corpus
corpus = convokit.Corpus(filename=convokit.download("supreme-corpus-v2"))

# compute coordination scores on this corpus
coord = convokit.Coordination()
coord.fit(corpus)

# get coordination scores
coord.transform(corpus)

# get set of all justices
justices = corpus.iter_users(lambda user: user.meta["is-justice"])
# get set of all users
everyone = corpus.iter_users()

# compute coordination from each justice to everyone
print("Justices, ranked by how much they coordinate to others:")
justices_to_everyone = coord.score(corpus, justices, everyone)
for justice, score in sorted(justices_to_everyone.averages_by_user().items(),
    key=lambda x: x[1], reverse=True):
    print(justice.name, round(score, 5))
print()

# compute coordination from everyone to each justice
print("Justices, ranked by how much others coordinate to them:")
everyone_to_justices = coord.score(corpus, everyone, justices, focus="targets")
for justice, score in sorted(everyone_to_justices.averages_by_user().items(), 
    key=lambda x: x[1], reverse=True):
    print(justice.name, round(score, 5))
print()
