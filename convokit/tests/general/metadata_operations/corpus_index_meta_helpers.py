from convokit import Corpus, Utterance, Speaker


def get_basic_corpus():
    return Corpus(
        utterances=[
            Utterance(
                id="0",
                text="hello world",
                conversation_id="convo_id_0",
                speaker=Speaker(id="alice"),
            ),
            Utterance(
                id="1",
                reply_to="0",
                text="my name is bob",
                conversation_id="convo_id_0",
                speaker=Speaker(id="bob"),
            ),
            Utterance(
                id="2",
                reply_to="1",
                text="this is a test",
                conversation_id="convo_id_0",
                speaker=Speaker(id="charlie"),
            ),
        ]
    )
