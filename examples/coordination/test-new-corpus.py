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

corpus.meta["test-bin-feature"] = [
    list(corpus.iter_utterances())[0].meta["parsed"],
    list(corpus.iter_utterances())[1].meta["parsed"]
]

for i, ut in enumerate(list(corpus.iter_utterances())):
    if i % 2 == 0:
        ut.meta["test-ut-meta"] = []
    else:
        ut.meta["test-ut-meta"] = [
            list(corpus.iter_utterances())[0].meta["parsed"],
            list(corpus.iter_utterances())[1].meta["parsed"]
        ]

for i, ut in enumerate(list(corpus.iter_utterances())):
    print(ut.meta)

print()
print("Dumping this parsed corpus into a folder test-dump-bin")
corpus.dump("test-dump-bin", base_path=".")

corpus_2 = convokit.Corpus(filename="test-dump-bin")
print()
print("Retrieving parse info from saved corpus dump")
for ut in list(corpus_2.iter_utterances())[:5]:
    for token in ut.meta["parsed"]:
        print("[%s %s]" % (token.tag_, token.text), end=' ')
    print()

print()
print("Utterance meta index:")
print(convokit.meta_index(corpus_2))

# make sure binary meta loaded correctly
for ut in list(corpus_2.iter_utterances()):
    assert(str(sorted(corpus.get_utterance(ut.id).meta.items())) ==
        str(sorted(ut.meta.items())))

corpus_3 = convokit.Corpus(filename="test-dump-bin", exclude_utterance_meta=["parsed"])
print("Testing exclude function. Meta when excluding 'parsed' field:")
print(convokit.meta_index(corpus=corpus_3))
print(convokit.meta_index(filename="test-dump-bin"))
