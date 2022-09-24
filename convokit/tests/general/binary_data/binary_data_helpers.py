from convokit import Corpus, Utterance, Speaker


def construct_corpus_with_binary_data():
    speaker_byte_arr1 = bytearray([120, 3, 255, 0, 100])
    speaker_byte_arr2 = bytearray([110, 3, 255, 90])
    utt_byte_arr1 = bytearray([99, 44, 33])
    utt_byte_arr2 = bytearray([110, 200, 220, 28])

    corpus = Corpus(
        utterances=[
            Utterance(
                id="0",
                text="hello world",
                speaker=Speaker(
                    id="alice", meta={"speaker_binary_data": speaker_byte_arr1, "index": 99}
                ),
                meta={"utt_binary_data": utt_byte_arr1},
            ),
            Utterance(
                id="1",
                text="my name is bob",
                speaker=Speaker(id="bob", meta={"speaker_binary_data": speaker_byte_arr2}),
                meta={"utt_binary_data": utt_byte_arr2},
            ),
            Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
        ]
    )
    return corpus
