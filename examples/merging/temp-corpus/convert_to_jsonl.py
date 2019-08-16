import os

def convert_json_to_jsonl(corpus_filepath):
    with open(filepath, 'r') as f:
        json_str = f.read()

    json_str = json_str[1:-1] #strip [ ]

    jsonl_filepath = filepath.replace(".json", ".jsonl")

    with open(jsonl_filepath, 'w') as f:
        bracket_ctr = 0
        utt_acc = ''
        ignore_chars = 0

        for c in json_str:
            if ignore_chars > 0:
                ignore_chars -= 1
                continue

            utt_acc += c

            if c == '{':
                bracket_ctr += 1
            elif c == '}':
                bracket_ctr -= 1
                if bracket_ctr == 0:
                    f.write(utt_acc)
                    f.write("\n")
                    utt_acc = ''
                    ignore_chars = 2



if __name__ == "__main__":
    convert_json_to_jsonl("./utterances.json")
    os.chdir("../../..")
    from convokit import Corpus
    os.chdir("./examples/merging/temp-corpus")

    corpus1 = Corpus(filename="utterances.jsonl")
    print(corpus1.utterances)

