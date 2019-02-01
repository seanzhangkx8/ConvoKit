import convokit

# load set of files: supreme.json and supreme-users.json
corpus = convokit.Corpus(filename=convokit.download("supreme-corpus-v2"))
print("users:")
print(corpus.get_usernames())
print()

corpus.dump("test-dump")

print("female users:")
for user in corpus.iter_users():
    if user.meta["gender"].lower().startswith("f"):
        print(user.name)

print("some utterances from a case:")
for ut in list(corpus.iter_utterances())[:5]:
    if ut.meta["case"] == "02-1472":
        print(ut.text)

print("testing Parser...")
parser = convokit.Parser()
corpus = parser.fit_transform(corpus)
print("POS-tagging results:")
for ut in list(corpus.iter_utterances())[:5]:
    for token in ut.meta["parsed"]:
        print("[%s %s]" % (token.tag_, token.text), end=' ')
    print()
