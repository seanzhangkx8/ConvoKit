import sys
import convokit
from collections import defaultdict

corpus = convokit.Corpus("../../datasets/reddit-corpus/reddit-convos.json")
threads = corpus.utterance_threads(prefix_len=10)

def disp(thread, root, indent=0):
    print(" "*indent + thread[root].user.name + ": " +
        thread[root].text.replace("\n", " "))
    children = [k for k, v in thread.items() if v.reply_to == root]
    for child in children:
        disp(thread, child, indent=indent+4)

if len(sys.argv) > 1:
    for root in sys.argv[1:]:
        print("--- {} ---".format(root))
        disp(threads[root], root)
        print()
else:
    while True:
        print("Enter thread root ID [t1_XXX]: ", end="")
        root = input()
        print("--- {} ---".format(root))
        disp(threads[root], root)
        print()
