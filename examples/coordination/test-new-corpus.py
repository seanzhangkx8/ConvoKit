import convokit

# load set of files: supreme.json and supreme-users.json
corpus = convokit.Corpus(filename="supreme.json")
print("users:")
print(corpus.get_usernames())
print()

print("female users:")
for user in corpus.iter_users():
    if user.meta["gender"].lower().startswith("f"):
        print(user.name)

print("some utterances from a case:")
for ut in list(corpus.iter_utterances())[:5]:
    if ut.meta["case"] == "02-1472":
        print(ut.text)
