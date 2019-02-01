import convokit

#corpus = convokit.Corpus(filename=convokit.download("supreme-corpus-v2"))
corpus = convokit.Corpus(utterances=[
    convokit.Utterance(id=0, text="hello world", user=convokit.User(name="bob")),
    convokit.Utterance(id=1, text="my name is bob", user=convokit.User(name="bob")),
    convokit.Utterance(id=2, text="this is a test", user=convokit.User(name="bob")),
    convokit.Utterance(id=3, text="i like pie", user=convokit.User(name="candice")),
    convokit.Utterance(id=4, text="this is a sentence", user=convokit.User(name="candice")),
    convokit.Utterance(id=5, text="goodbye", user=convokit.User(name="candice")),
    ])
print("users:")
print(corpus.get_usernames())
print()

#print("female users:")
#for user in corpus.iter_users():
#    if user.meta["gender"].lower().startswith("f"):
#        print(user.name)
#
#print("some utterances from a case:")
#for ut in list(corpus.iter_utterances())[:5]:
#    if ut.meta["case"] == "02-1472":
#        print(ut.text)

print("testing Parser...")
parser = convokit.Parser()
corpus = parser.fit_transform(corpus)
print("POS-tagging results:")
for ut in list(corpus.iter_utterances())[:5]:
    for token in ut.meta["parsed"]:
        print("[%s %s]" % (token.tag_, token.text), end=' ')
    print()

print()
print("Dumping this parsed corpus into a folder test-dump-bin")
corpus.dump("test-dump-bin")

corpus_2 = convokit.Corpus(filename="test-dump-bin")
print()
print("Retrieving parse info from saved corpus dump")
for ut in list(corpus_2.iter_utterances())[:5]:
    for token in ut.meta["parsed"]:
        print("[%s %s]" % (token.tag_, token.text), end=' ')
    print()
