def processDialog(voc, dialog):
    processed = []
    for utterance in dialog.iter_utterances():
        # skip the section header, which does not contain conversational content
        # TODO exclude specific utterances
        # if utterance.meta['is_section_header']:
        #     continue
        tokens = tokenize(utterance.text)
        # replace out-of-vocabulary tokens
        for i in range(len(tokens)):
            if tokens[i] not in voc.word2index:
                tokens[i] = "UNK"
        processed.append({"tokens": tokens, "id": utterance.id})
        # processed.append({"tokens": tokens, "is_attack": int(utterance.meta['comment_has_personal_attack']), "id": utterance.id})
    return processed