import os
os.chdir("../..")
from convokit import Corpus
os.chdir("./examples/merging")

def convert_json_to_json(corpus_filepath):
    corpus = Corpus(filename=corpus_filepath)
    corpus.dump(name="temp-corpus2", save_to_existing_path=True)
#
# def convert_json_to_jsonl(filepath):
#     with open(filepath, 'r') as f:
#         json_str = f.read()
#
#     json_str = json_str[1:-1] #strip [ ]
#
#     jsonl_filepath = filepath.replace(".json", ".jsonl")
#
#     with open(jsonl_filepath, 'w') as f:
#         bracket_ctr = 0
#         utt_acc = ''
#         ignore_chars = 0
#
#         for c in json_str:
#             if ignore_chars > 0:
#                 ignore_chars -= 1
#                 continue
#
#             utt_acc += c
#
#             if c == '{':
#                 bracket_ctr += 1
#             elif c == '}':
#                 bracket_ctr -= 1
#                 if bracket_ctr == 0:
#                     f.write(utt_acc)
#                     f.write("\n")
#                     utt_acc = ''
#                     ignore_chars = 2


if __name__ == "__main__":
    convert_json_to_jsonl("temp-corpus")

    corpus1 = Corpus(filename="temp-corpus2")
    print(corpus1.utterances)

