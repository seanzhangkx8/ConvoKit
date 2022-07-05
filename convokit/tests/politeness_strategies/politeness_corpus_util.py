from convokit.model import Corpus, Utterance, Speaker

# test utterances
GRATITUDE = "I really appreciate that you have done them. Thanks a lot."
DEFERENCE = "Nice work so far on your rewrite."
GREETING = "Hey, how are you doing these days?"
APOLOGY = "Sorry to bother you, but I need someone to work on this project."
PLEASE = "Could you please elaborate more?"
PLEASE_START = "Please do not remove warnings."
BTW = "By the way, where did you find that picture?"
DIRECT_QN = "What is your native language?"
DIRECT_START = "So can you retrieve it or not?"
SUBJUNCTIVE = "Could you please proofread this article?"
INDICATIVE = "Can you proofread this article for me?"
HEDGES = "I suggest we start with the simplest case."
FACTUALITY = "In fact, our data does not support this claim."

GRATITUDE_ZH = "非常感谢您的帮助。"
DEFERENCE_ZH = "干得漂亮!改写得非常好。"
GREETING_ZH = "嗨，你现在有空吗？"
APOLOGY_ZH = "不好意思打扰你了，你有时间帮我修改一下这份草稿吗？"
PLEASE_ZH = "可不可以请您说慢一点？"
PLEASE_START_ZH = "请留意我们的后续通知。"
BTW_ZH = "顺便问一下，你是在哪里看到这些照片的？"
DIRECT_QN_ZH = "哪里出了问题？"
HEDGES_ZH = "应该可以吧。"
FACTUALITY_ZH = "说实话，我也不懂这些。"


def politeness_test_corpus():
    speakers = [Speaker(id="alice"), Speaker(id="bob")]
    texts = [
        GRATITUDE,
        DEFERENCE,
        GREETING,
        APOLOGY,
        PLEASE,
        PLEASE_START,
        BTW,
        DIRECT_QN,
        DIRECT_START,
        SUBJUNCTIVE,
        INDICATIVE,
        HEDGES,
        FACTUALITY,
    ]

    utterances = [Utterance(id="0", text=texts[0], speaker=speakers[1], reply_to=None)]
    for i, text in enumerate(texts[1:]):
        utterances.append(
            Utterance(id=str(i + 1), text=text, speaker=speakers[i % 2], reply_to=str(i))
        )

    return Corpus(utterances=utterances)


def politeness_test_zh_corpus():
    speakers = [Speaker(id="alice"), Speaker(id="bob")]
    texts = [
        GRATITUDE_ZH,
        DEFERENCE_ZH,
        GREETING_ZH,
        APOLOGY_ZH,
        PLEASE_ZH,
        PLEASE_START_ZH,
        BTW_ZH,
        DIRECT_QN_ZH,
        HEDGES_ZH,
        FACTUALITY_ZH,
    ]

    utterances = [Utterance(id="0", text=texts[0], speaker=speakers[1], reply_to=None)]
    for i, text in enumerate(texts[1:]):
        utterances.append(
            Utterance(id=str(i + 1), text=text, speaker=speakers[i % 2], reply_to=str(i))
        )

    return Corpus(utterances=utterances)


def parsed_politeness_test_corpus():
    corpus = politeness_test_corpus()
    parses = {
        "0": [
            {
                "rt": 2,
                "toks": [
                    {"tok": "I", "tag": "PRP", "dep": "nsubj", "up": 2, "dn": []},
                    {"tok": "really", "tag": "RB", "dep": "advmod", "up": 2, "dn": []},
                    {"tok": "appreciate", "tag": "VBP", "dep": "ROOT", "dn": [0, 1, 6, 8]},
                    {"tok": "that", "tag": "IN", "dep": "mark", "up": 6, "dn": []},
                    {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 6, "dn": []},
                    {"tok": "have", "tag": "VBP", "dep": "aux", "up": 6, "dn": []},
                    {"tok": "done", "tag": "VBN", "dep": "ccomp", "up": 2, "dn": [3, 4, 5, 7]},
                    {"tok": "them", "tag": "PRP", "dep": "dobj", "up": 6, "dn": []},
                    {"tok": ".", "tag": ".", "dep": "punct", "up": 2, "dn": []},
                ],
            },
            {
                "rt": 0,
                "toks": [
                    {"tok": "Thanks", "tag": "NNS", "dep": "ROOT", "dn": [2, 3]},
                    {"tok": "a", "tag": "DT", "dep": "det", "up": 2, "dn": []},
                    {"tok": "lot", "tag": "NN", "dep": "npadvmod", "up": 0, "dn": [1]},
                    {"tok": ".", "tag": ".", "dep": "punct", "up": 0, "dn": []},
                ],
            },
        ],
        "1": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "Nice", "tag": "JJ", "dep": "amod", "up": 1, "dn": []},
                    {"tok": "work", "tag": "NN", "dep": "ROOT", "dn": [0, 3, 4, 7]},
                    {"tok": "so", "tag": "RB", "dep": "advmod", "up": 3, "dn": []},
                    {"tok": "far", "tag": "RB", "dep": "advmod", "up": 1, "dn": [2]},
                    {"tok": "on", "tag": "IN", "dep": "prep", "up": 1, "dn": [6]},
                    {"tok": "your", "tag": "PRP$", "dep": "poss", "up": 6, "dn": []},
                    {"tok": "rewrite", "tag": "NN", "dep": "pobj", "up": 4, "dn": [5]},
                    {"tok": ".", "tag": ".", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "2": [
            {
                "rt": 5,
                "toks": [
                    {"tok": "Hey", "tag": "UH", "dep": "intj", "up": 5, "dn": []},
                    {"tok": ",", "tag": ",", "dep": "punct", "up": 5, "dn": []},
                    {"tok": "how", "tag": "WRB", "dep": "advmod", "up": 5, "dn": []},
                    {"tok": "are", "tag": "VBP", "dep": "aux", "up": 5, "dn": []},
                    {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 5, "dn": []},
                    {"tok": "doing", "tag": "VBG", "dep": "ROOT", "dn": [0, 1, 2, 3, 4, 7, 8]},
                    {"tok": "these", "tag": "DT", "dep": "det", "up": 7, "dn": []},
                    {"tok": "days", "tag": "NNS", "dep": "dobj", "up": 5, "dn": [6]},
                    {"tok": "?", "tag": ".", "dep": "punct", "up": 5, "dn": []},
                ],
            }
        ],
        "3": [
            {
                "rt": 0,
                "toks": [
                    {"tok": "Sorry", "tag": "JJ", "dep": "ROOT", "dn": [2, 4, 5, 7]},
                    {"tok": "to", "tag": "TO", "dep": "aux", "up": 2, "dn": []},
                    {"tok": "bother", "tag": "VB", "dep": "xcomp", "up": 0, "dn": [1, 3]},
                    {"tok": "you", "tag": "PRP", "dep": "dobj", "up": 2, "dn": []},
                    {"tok": ",", "tag": ",", "dep": "punct", "up": 0, "dn": []},
                    {"tok": "but", "tag": "CC", "dep": "cc", "up": 0, "dn": []},
                    {"tok": "I", "tag": "PRP", "dep": "nsubj", "up": 7, "dn": []},
                    {"tok": "need", "tag": "VBP", "dep": "conj", "up": 0, "dn": [6, 8, 14]},
                    {"tok": "someone", "tag": "NN", "dep": "dobj", "up": 7, "dn": [10]},
                    {"tok": "to", "tag": "TO", "dep": "aux", "up": 10, "dn": []},
                    {"tok": "work", "tag": "VB", "dep": "relcl", "up": 8, "dn": [9, 11]},
                    {"tok": "on", "tag": "IN", "dep": "prep", "up": 10, "dn": [13]},
                    {"tok": "this", "tag": "DT", "dep": "det", "up": 13, "dn": []},
                    {"tok": "project", "tag": "NN", "dep": "pobj", "up": 11, "dn": [12]},
                    {"tok": ".", "tag": ".", "dep": "punct", "up": 7, "dn": []},
                ],
            }
        ],
        "4": [
            {
                "rt": 3,
                "toks": [
                    {"tok": "Could", "tag": "MD", "dep": "aux", "up": 3, "dn": []},
                    {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 3, "dn": []},
                    {"tok": "please", "tag": "UH", "dep": "intj", "up": 3, "dn": []},
                    {"tok": "elaborate", "tag": "VB", "dep": "ROOT", "dn": [0, 1, 2, 4, 5]},
                    {"tok": "more", "tag": "RBR", "dep": "advmod", "up": 3, "dn": []},
                    {"tok": "?", "tag": ".", "dep": "punct", "up": 3, "dn": []},
                ],
            }
        ],
        "5": [
            {
                "rt": 3,
                "toks": [
                    {"tok": "Please", "tag": "UH", "dep": "intj", "up": 3, "dn": []},
                    {"tok": "do", "tag": "VB", "dep": "aux", "up": 3, "dn": []},
                    {"tok": "not", "tag": "RB", "dep": "neg", "up": 3, "dn": []},
                    {"tok": "remove", "tag": "VB", "dep": "ROOT", "dn": [0, 1, 2, 4, 5]},
                    {"tok": "warnings", "tag": "NNS", "dep": "dobj", "up": 3, "dn": []},
                    {"tok": ".", "tag": ".", "dep": "punct", "up": 3, "dn": []},
                ],
            }
        ],
        "6": [
            {
                "rt": 7,
                "toks": [
                    {"tok": "By", "tag": "IN", "dep": "prep", "up": 7, "dn": [2]},
                    {"tok": "the", "tag": "DT", "dep": "det", "up": 2, "dn": []},
                    {"tok": "way", "tag": "NN", "dep": "pobj", "up": 0, "dn": [1]},
                    {"tok": ",", "tag": ",", "dep": "punct", "up": 7, "dn": []},
                    {"tok": "where", "tag": "WRB", "dep": "advmod", "up": 7, "dn": []},
                    {"tok": "did", "tag": "VBD", "dep": "aux", "up": 7, "dn": []},
                    {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 7, "dn": []},
                    {"tok": "find", "tag": "VB", "dep": "ROOT", "dn": [0, 3, 4, 5, 6, 9, 10]},
                    {"tok": "that", "tag": "DT", "dep": "det", "up": 9, "dn": []},
                    {"tok": "picture", "tag": "NN", "dep": "dobj", "up": 7, "dn": [8]},
                    {"tok": "?", "tag": ".", "dep": "punct", "up": 7, "dn": []},
                ],
            }
        ],
        "7": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "What", "tag": "WP", "dep": "attr", "up": 1, "dn": []},
                    {"tok": "is", "tag": "VBZ", "dep": "ROOT", "dn": [0, 4, 5]},
                    {"tok": "your", "tag": "PRP$", "dep": "poss", "up": 4, "dn": []},
                    {"tok": "native", "tag": "JJ", "dep": "amod", "up": 4, "dn": []},
                    {"tok": "language", "tag": "NN", "dep": "nsubj", "up": 1, "dn": [2, 3]},
                    {"tok": "?", "tag": ".", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "8": [
            {
                "rt": 3,
                "toks": [
                    {"tok": "So", "tag": "RB", "dep": "advmod", "up": 3, "dn": []},
                    {"tok": "can", "tag": "MD", "dep": "aux", "up": 3, "dn": []},
                    {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 3, "dn": []},
                    {"tok": "retrieve", "tag": "VB", "dep": "ROOT", "dn": [0, 1, 2, 4, 5, 6, 7]},
                    {"tok": "it", "tag": "PRP", "dep": "dobj", "up": 3, "dn": []},
                    {"tok": "or", "tag": "CC", "dep": "cc", "up": 3, "dn": []},
                    {"tok": "not", "tag": "RB", "dep": "conj", "up": 3, "dn": []},
                    {"tok": "?", "tag": ".", "dep": "punct", "up": 3, "dn": []},
                ],
            }
        ],
        "9": [
            {
                "rt": 3,
                "toks": [
                    {"tok": "Could", "tag": "MD", "dep": "aux", "up": 3, "dn": []},
                    {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 3, "dn": []},
                    {"tok": "please", "tag": "UH", "dep": "intj", "up": 3, "dn": []},
                    {"tok": "proofread", "tag": "VB", "dep": "ROOT", "dn": [0, 1, 2, 5, 6]},
                    {"tok": "this", "tag": "DT", "dep": "det", "up": 5, "dn": []},
                    {"tok": "article", "tag": "NN", "dep": "dobj", "up": 3, "dn": [4]},
                    {"tok": "?", "tag": ".", "dep": "punct", "up": 3, "dn": []},
                ],
            }
        ],
        "10": [
            {
                "rt": 2,
                "toks": [
                    {"tok": "Can", "tag": "MD", "dep": "aux", "up": 2, "dn": []},
                    {"tok": "you", "tag": "PRP", "dep": "nsubj", "up": 2, "dn": []},
                    {"tok": "proofread", "tag": "VB", "dep": "ROOT", "dn": [0, 1, 4, 5, 7]},
                    {"tok": "this", "tag": "DT", "dep": "det", "up": 4, "dn": []},
                    {"tok": "article", "tag": "NN", "dep": "dobj", "up": 2, "dn": [3]},
                    {"tok": "for", "tag": "IN", "dep": "dative", "up": 2, "dn": [6]},
                    {"tok": "me", "tag": "PRP", "dep": "pobj", "up": 5, "dn": []},
                    {"tok": "?", "tag": ".", "dep": "punct", "up": 2, "dn": []},
                ],
            }
        ],
        "11": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "I", "tag": "PRP", "dep": "nsubj", "up": 1, "dn": []},
                    {"tok": "suggest", "tag": "VBP", "dep": "ROOT", "dn": [0, 3, 8]},
                    {"tok": "we", "tag": "PRP", "dep": "nsubj", "up": 3, "dn": []},
                    {"tok": "start", "tag": "VBP", "dep": "ccomp", "up": 1, "dn": [2, 4]},
                    {"tok": "with", "tag": "IN", "dep": "prep", "up": 3, "dn": [7]},
                    {"tok": "the", "tag": "DT", "dep": "det", "up": 7, "dn": []},
                    {"tok": "simplest", "tag": "JJS", "dep": "amod", "up": 7, "dn": []},
                    {"tok": "case", "tag": "NN", "dep": "pobj", "up": 4, "dn": [5, 6]},
                    {"tok": ".", "tag": ".", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "12": [
            {
                "rt": 7,
                "toks": [
                    {"tok": "In", "tag": "IN", "dep": "prep", "up": 7, "dn": [1]},
                    {"tok": "fact", "tag": "NN", "dep": "pobj", "up": 0, "dn": []},
                    {"tok": ",", "tag": ",", "dep": "punct", "up": 7, "dn": []},
                    {"tok": "our", "tag": "PRP$", "dep": "poss", "up": 4, "dn": []},
                    {"tok": "data", "tag": "NNS", "dep": "nsubj", "up": 7, "dn": [3]},
                    {"tok": "does", "tag": "VBZ", "dep": "aux", "up": 7, "dn": []},
                    {"tok": "not", "tag": "RB", "dep": "neg", "up": 7, "dn": []},
                    {"tok": "support", "tag": "VB", "dep": "ROOT", "dn": [0, 2, 4, 5, 6, 9, 10]},
                    {"tok": "this", "tag": "DT", "dep": "det", "up": 9, "dn": []},
                    {"tok": "claim", "tag": "NN", "dep": "dobj", "up": 7, "dn": [8]},
                    {"tok": ".", "tag": ".", "dep": "punct", "up": 7, "dn": []},
                ],
            }
        ],
    }

    for idx, parse in parses.items():
        corpus.get_utterance(idx).add_meta("parsed", parse)

    return corpus


def parsed_politeness_test_zh_corpus():
    corpus = politeness_test_zh_corpus()
    parses = {
        "0": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "非常", "tag": "AD", "dep": "advmod", "up": 1, "dn": []},
                    {"tok": "感谢", "tag": "VV", "dep": "ROOT", "dn": [0, 4, 5]},
                    {"tok": "您", "tag": "PN", "dep": "nmod:assmod", "up": 4, "dn": [3]},
                    {"tok": "的", "tag": "DEG", "dep": "case", "up": 2, "dn": []},
                    {"tok": "帮助", "tag": "NN", "dep": "dobj", "up": 1, "dn": [2]},
                    {"tok": "。", "tag": "PU", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "1": [
            {
                "rt": 7,
                "toks": [
                    {"tok": "干", "tag": "VV", "dep": "dep", "up": 2, "dn": []},
                    {"tok": "得", "tag": "DER", "dep": "dep", "up": 2, "dn": []},
                    {"tok": "漂亮", "tag": "VA", "dep": "dep", "up": 7, "dn": [0, 1]},
                    {"tok": "!改", "tag": "AD", "dep": "advmod", "up": 7, "dn": []},
                    {"tok": "写", "tag": "VV", "dep": "dep", "up": 7, "dn": []},
                    {"tok": "得", "tag": "DER", "dep": "dep", "up": 7, "dn": []},
                    {"tok": "非常", "tag": "AD", "dep": "advmod", "up": 7, "dn": []},
                    {"tok": "好", "tag": "VA", "dep": "ROOT", "dn": [2, 3, 4, 5, 6, 8]},
                    {"tok": "。", "tag": "PU", "dep": "punct", "up": 7, "dn": []},
                ],
            }
        ],
        "2": [
            {
                "rt": 6,
                "toks": [
                    {"tok": "嗨", "tag": "IJ", "dep": "dep", "up": 4, "dn": []},
                    {"tok": "，", "tag": "PU", "dep": "punct", "up": 4, "dn": []},
                    {"tok": "你", "tag": "PN", "dep": "nsubj", "up": 4, "dn": []},
                    {"tok": "现在", "tag": "NT", "dep": "nmod:tmod", "up": 4, "dn": []},
                    {"tok": "有空", "tag": "VV", "dep": "dep", "up": 6, "dn": [0, 1, 2, 3]},
                    {"tok": "吗", "tag": "SP", "dep": "discourse", "up": 6, "dn": []},
                    {"tok": "？", "tag": "PU", "dep": "ROOT", "dn": [4, 5]},
                ],
            }
        ],
        "3": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "不好意思", "tag": "AD", "dep": "advmod", "up": 1, "dn": []},
                    {"tok": "打扰", "tag": "VV", "dep": "ROOT", "dn": [0, 2, 3, 4]},
                    {"tok": "你", "tag": "PN", "dep": "dobj", "up": 1, "dn": []},
                    {"tok": "了", "tag": "SP", "dep": "discourse", "up": 1, "dn": []},
                    {"tok": "，", "tag": "PU", "dep": "punct", "up": 1, "dn": []},
                ],
            },
            {
                "rt": 11,
                "toks": [
                    {"tok": "你", "tag": "PN", "dep": "dep", "up": 1, "dn": []},
                    {"tok": "有", "tag": "VE", "dep": "dep", "up": 11, "dn": [0, 2, 5]},
                    {"tok": "时间", "tag": "NN", "dep": "dobj", "up": 1, "dn": []},
                    {"tok": "帮", "tag": "P", "dep": "case", "up": 4, "dn": []},
                    {"tok": "我", "tag": "PN", "dep": "nmod:prep", "up": 5, "dn": [3]},
                    {"tok": "修改", "tag": "VV", "dep": "conj", "up": 1, "dn": [4, 6, 9]},
                    {"tok": "一下", "tag": "AD", "dep": "advmod", "up": 5, "dn": []},
                    {"tok": "这", "tag": "DT", "dep": "det", "up": 9, "dn": [8]},
                    {"tok": "份", "tag": "M", "dep": "mark:clf", "up": 7, "dn": []},
                    {"tok": "草稿", "tag": "NN", "dep": "dobj", "up": 5, "dn": [7]},
                    {"tok": "吗", "tag": "SP", "dep": "discourse", "up": 11, "dn": []},
                    {"tok": "？", "tag": "PU", "dep": "ROOT", "dn": [1, 10]},
                ],
            },
        ],
        "4": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "可不可以", "tag": "AD", "dep": "advmod", "up": 1, "dn": []},
                    {"tok": "请", "tag": "VV", "dep": "ROOT", "dn": [0, 2, 3, 6]},
                    {"tok": "您", "tag": "PN", "dep": "dobj", "up": 1, "dn": []},
                    {"tok": "说", "tag": "VV", "dep": "ccomp", "up": 1, "dn": [4]},
                    {"tok": "慢", "tag": "VA", "dep": "ccomp", "up": 3, "dn": [5]},
                    {"tok": "一点", "tag": "AD", "dep": "advmod", "up": 4, "dn": []},
                    {"tok": "？", "tag": "PU", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "5": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "请", "tag": "VV", "dep": "xcomp", "up": 1, "dn": []},
                    {"tok": "留意", "tag": "VV", "dep": "ROOT", "dn": [0, 5, 6]},
                    {"tok": "我们", "tag": "PN", "dep": "nmod:assmod", "up": 5, "dn": [3]},
                    {"tok": "的", "tag": "DEG", "dep": "case", "up": 2, "dn": []},
                    {"tok": "后续", "tag": "JJ", "dep": "amod", "up": 5, "dn": []},
                    {"tok": "通知", "tag": "NN", "dep": "dobj", "up": 1, "dn": [2, 4]},
                    {"tok": "。", "tag": "PU", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "6": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "顺便", "tag": "AD", "dep": "advmod", "up": 1, "dn": []},
                    {"tok": "问", "tag": "VV", "dep": "ROOT", "dn": [0, 2, 3, 8, 12]},
                    {"tok": "一下", "tag": "AD", "dep": "advmod", "up": 1, "dn": []},
                    {"tok": "，", "tag": "PU", "dep": "punct", "up": 1, "dn": []},
                    {"tok": "你", "tag": "PN", "dep": "nsubj", "up": 8, "dn": []},
                    {"tok": "是", "tag": "VC", "dep": "cop", "up": 8, "dn": []},
                    {"tok": "在", "tag": "P", "dep": "case", "up": 7, "dn": []},
                    {"tok": "哪里", "tag": "PN", "dep": "nmod:prep", "up": 8, "dn": [6]},
                    {"tok": "看到", "tag": "VV", "dep": "dep", "up": 1, "dn": [4, 5, 7, 10, 11]},
                    {"tok": "这些", "tag": "DT", "dep": "det", "up": 10, "dn": []},
                    {"tok": "照片", "tag": "NN", "dep": "dobj", "up": 8, "dn": [9]},
                    {"tok": "的", "tag": "SP", "dep": "discourse", "up": 8, "dn": []},
                    {"tok": "？", "tag": "PU", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "7": [
            {
                "rt": 1,
                "toks": [
                    {"tok": "哪里", "tag": "PN", "dep": "nsubj", "up": 1, "dn": []},
                    {"tok": "出", "tag": "VV", "dep": "ROOT", "dn": [0, 2, 3, 4]},
                    {"tok": "了", "tag": "AS", "dep": "aux:asp", "up": 1, "dn": []},
                    {"tok": "问题", "tag": "NN", "dep": "dobj", "up": 1, "dn": []},
                    {"tok": "？", "tag": "PU", "dep": "punct", "up": 1, "dn": []},
                ],
            }
        ],
        "8": [
            {
                "rt": 3,
                "toks": [
                    {"tok": "应该", "tag": "VV", "dep": "aux:modal", "up": 1, "dn": []},
                    {"tok": "可以", "tag": "VV", "dep": "dep", "up": 3, "dn": [0]},
                    {"tok": "吧", "tag": "SP", "dep": "discourse", "up": 3, "dn": []},
                    {"tok": "。", "tag": "PU", "dep": "ROOT", "dn": [1, 2]},
                ],
            }
        ],
        "9": [
            {
                "rt": 6,
                "toks": [
                    {"tok": "说", "tag": "VV", "dep": "dep", "up": 6, "dn": [1]},
                    {"tok": "实话", "tag": "NN", "dep": "dobj", "up": 0, "dn": []},
                    {"tok": "，", "tag": "PU", "dep": "punct", "up": 6, "dn": []},
                    {"tok": "我", "tag": "PN", "dep": "nsubj", "up": 6, "dn": []},
                    {"tok": "也", "tag": "AD", "dep": "advmod", "up": 6, "dn": []},
                    {"tok": "不", "tag": "AD", "dep": "neg", "up": 6, "dn": []},
                    {"tok": "懂", "tag": "VV", "dep": "ROOT", "dn": [0, 2, 3, 4, 5, 7, 8]},
                    {"tok": "这些", "tag": "PN", "dep": "dobj", "up": 6, "dn": []},
                    {"tok": "。", "tag": "PU", "dep": "punct", "up": 6, "dn": []},
                ],
            }
        ],
    }

    for idx, parse in parses.items():
        corpus.get_utterance(idx).add_meta("parsed", parse)

    return corpus
