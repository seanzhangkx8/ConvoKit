"""
Contains functions that help with the construction / dumping of a Corpus
"""

import json
import os
import pickle
from collections import defaultdict
from typing import Dict, Optional, List

import bson
from pymongo import UpdateOne

from convokit.util import warn
from .conversation import Conversation
from .convoKitIndex import ConvoKitIndex
from .convoKitMeta import ConvoKitMeta
from .speaker import Speaker
from .storageManager import StorageManager, MemStorageManager, DBStorageManager
from .utterance import Utterance

BIN_DELIM_L, BIN_DELIM_R = "<##bin{", "}&&@**>"
KeyId = "id"
KeySpeaker = "speaker"
KeyConvoId = "conversation_id"
KeyReplyTo = "reply-to"
KeyTimestamp = "timestamp"
KeyText = "text"
DefinedKeys = {KeyId, KeySpeaker, KeyConvoId, KeyReplyTo, KeyTimestamp, KeyText}
KeyMeta = "meta"
KeyVectors = "vectors"

JSONLIST_BUFFER_SIZE = 1000


def get_corpus_id(db_collection_prefix: Optional[str], filename: Optional[str]) -> Optional[str]:
    if db_collection_prefix is not None:
        # treat the unique collection prefix as the ID (even if a filename is specified)
        return db_collection_prefix
    elif filename is not None:
        # automatically derive an ID from the file path
        return os.path.basename(os.path.normpath(filename))
    else:
        return None


def get_corpus_dirpath(filename: str) -> Optional[str]:
    if filename is None:
        return None
    elif os.path.isdir(filename):
        return filename
    else:
        return os.path.dirname(filename)


def initialize_storage(
    corpus: "Corpus", storage: Optional[StorageManager], storage_type: str, db_host: Optional[str]
):
    if storage is not None:
        return storage
    else:
        if storage_type == "mem":
            return MemStorageManager()
        elif storage_type == "db":
            if db_host is None:
                db_host = corpus.config.db_host
            return DBStorageManager(corpus.id, db_host)
        else:
            raise ValueError(
                f"Unrecognized setting '{storage_type}' for storage type; should be either 'mem' or 'db'."
            )


def load_utterance_info_from_dir(
    dirname, utterance_start_index, utterance_end_index, exclude_utterance_meta
):
    assert dirname is not None
    assert os.path.isdir(dirname)

    if utterance_start_index is None:
        utterance_start_index = 0
    if utterance_end_index is None:
        utterance_end_index = float("inf")

    if os.path.exists(os.path.join(dirname, "utterances.jsonl")):
        with open(os.path.join(dirname, "utterances.jsonl"), "r") as f:
            utterances = []
            idx = 0
            for line in f:
                if utterance_start_index <= idx <= utterance_end_index:
                    utterances.append(json.loads(line))
                idx += 1

    elif os.path.exists(os.path.join(dirname, "utterances.json")):
        with open(os.path.join(dirname, "utterances.json"), "r") as f:
            utterances = json.load(f)

    if exclude_utterance_meta:
        for utt in utterances:
            for field in exclude_utterance_meta:
                del utt["meta"][field]

    return utterances


def load_speakers_data_from_dir(filename, exclude_speaker_meta):
    speaker_file = "speakers.json" if "speakers.json" in os.listdir(filename) else "users.json"
    with open(os.path.join(filename, speaker_file), "r") as f:
        id_to_speaker_data = json.load(f)

        if (
            len(id_to_speaker_data) > 0
            and len(next(iter(id_to_speaker_data.values())))
            and "vectors" in id_to_speaker_data == 2
        ):
            # has vectors data
            for _, speaker_data in id_to_speaker_data.items():
                for k in exclude_speaker_meta:
                    if k in speaker_data["meta"]:
                        del speaker_data["meta"][k]
        else:
            for _, speaker_data in id_to_speaker_data.items():
                for k in exclude_speaker_meta:
                    if k in speaker_data:
                        del speaker_data[k]
    return id_to_speaker_data


def load_convos_data_from_dir(filename, exclude_conversation_meta):
    """

    :param filename:
    :param exclude_conversation_meta:
    :return: a mapping from convo id to convo meta
    """
    with open(os.path.join(filename, "conversations.json"), "r") as f:
        id_to_convo_data = json.load(f)

        if (
            len(id_to_convo_data) > 0
            and len(next(iter(id_to_convo_data.values())))
            and "vectors" in id_to_convo_data == 2
        ):
            # has vectors data
            for _, convo_data in id_to_convo_data.items():
                for k in exclude_conversation_meta:
                    if k in convo_data["meta"]:
                        del convo_data["meta"][k]
        else:
            for _, convo_data in id_to_convo_data.items():
                for k in exclude_conversation_meta:
                    if k in convo_data:
                        del convo_data[k]
    return id_to_convo_data


def load_corpus_meta_from_dir(filename, corpus_meta, exclude_overall_meta):
    """
    Updates corpus meta object with fields from corpus.json
    """
    with open(os.path.join(filename, "corpus.json"), "r") as f:
        for k, v in json.load(f).items():
            if k in exclude_overall_meta:
                continue
            corpus_meta[k] = v


def unpack_binary_data_for_utts(utterances, filename, utterance_index, exclude_meta, KeyMeta):
    """

    :param utterances: mapping from utterance id to {'meta': ..., 'vectors': ...}
    :param filename: filepath containing corpus files
    :param utterance_index: utterance meta index
    :param exclude_meta: list of metadata attributes to exclude
    :param KeyMeta: name of metadata key, should be 'meta'
    :return:
    """
    for field, field_types in utterance_index.items():
        if field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-bin.p"), "rb") as f:
                l_bin = pickle.load(f)
            for i, ut in enumerate(utterances):
                for k, v in ut[KeyMeta].items():
                    if (
                        k == field
                        and type(v) == str
                        and v.startswith(BIN_DELIM_L)
                        and v.endswith(BIN_DELIM_R)
                    ):
                        idx = int(v[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        utterances[i][KeyMeta][k] = l_bin[idx]
    for field in exclude_meta:
        del utterance_index[field]


def unpack_binary_data(filename, objs_data, object_index, obj_type, exclude_meta):
    """
    Unpack binary data for Speakers or Conversations

    :param filename: filepath containing the corpus data
    :param objs_data: a mapping from object id to a dictionary with two keys: 'meta' and 'vectors';
        in older versions, this is a mapping from object id to the metadata dictionary
    :param object_index: the meta_index dictionary for the component type
    :param obj_type: object type (i.e. speaker or conversation)
    :param exclude_meta: list of metadata attributes to exclude
    :return: None (mutates objs_data)
    """
    """
    Unpack binary data for Speakers or Conversations

    """
    # unpack speaker meta
    for field, field_types in object_index.items():
        if field_types[0] == "bin" and field not in exclude_meta:
            with open(os.path.join(filename, field + "-{}-bin.p".format(obj_type)), "rb") as f:
                l_bin = pickle.load(f)
            for obj, data in objs_data.items():
                metadata = data["meta"] if len(data) == 2 and "vectors" in data else data
                for k, v in metadata.items():
                    if (
                        k == field
                        and type(v) == str
                        and str(v).startswith(BIN_DELIM_L)
                        and str(v).endswith(BIN_DELIM_R)
                    ):
                        idx = int(v[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        metadata[k] = l_bin[idx]
    for field in exclude_meta:
        del object_index[field]


def unpack_all_binary_data(
    filename: str,
    meta_index: ConvoKitIndex,
    meta: ConvoKitMeta,
    utterances: List[Utterance],
    speakers_data: Dict[str, Dict],
    convos_data: Dict[str, Dict],
    exclude_utterance_meta: List[str],
    exclude_speaker_meta: List[str],
    exclude_conversation_meta: List[str],
    exclude_overall_meta: List[str],
):
    # unpack binary data for utterances
    unpack_binary_data_for_utts(
        utterances,
        filename,
        meta_index.utterances_index,
        exclude_utterance_meta,
        KeyMeta,
    )
    # unpack binary data for speakers
    unpack_binary_data(
        filename,
        speakers_data,
        meta_index.speakers_index,
        "speaker",
        exclude_speaker_meta,
    )

    # unpack binary data for conversations
    unpack_binary_data(
        filename,
        convos_data,
        meta_index.conversations_index,
        "convo",
        exclude_conversation_meta,
    )

    # unpack binary data for overall corpus
    unpack_binary_data(
        filename,
        meta,
        meta_index.overall_index,
        "overall",
        exclude_overall_meta,
    )


def load_from_utterance_file(filename, utterance_start_index, utterance_end_index):
    """
    where filename is "utterances.json" or "utterances.jsonl" for example
    """
    with open(filename, "r") as f:
        try:
            ext = filename.split(".")[-1]
            if ext == "json":
                utterances = json.load(f)
            elif ext == "jsonl":
                utterances = []
                if utterance_start_index is None:
                    utterance_start_index = 0
                if utterance_end_index is None:
                    utterance_end_index = float("inf")
                idx = 0
                for line in f:
                    if utterance_start_index <= idx <= utterance_end_index:
                        utterances.append(json.loads(line))
                    idx += 1
        except Exception as e:
            raise Exception(
                "Could not load corpus. Expected json file, encountered error: \n" + str(e)
            )
    return utterances


def initialize_speakers_and_utterances_objects(corpus, utterances, speakers_data):
    """
    Initialize Speaker and Utterance objects
    """
    if len(utterances) > 0:  # utterances might be empty for invalid corpus start/end indices
        KeySpeaker = "speaker" if "speaker" in utterances[0] else "user"
        KeyConvoId = "conversation_id" if "conversation_id" in utterances[0] else "root"

    for i, u in enumerate(utterances):
        u = defaultdict(lambda: None, u)
        speaker_key = u[KeySpeaker]
        if speaker_key not in corpus.speakers:
            if u[KeySpeaker] not in speakers_data:
                warn(
                    "CorpusLoadWarning: Missing speaker metadata for speaker ID: {}. "
                    "Initializing default empty metadata instead.".format(u[KeySpeaker])
                )
                speakers_data[u[KeySpeaker]] = {}
            if KeyMeta in speakers_data[u[KeySpeaker]]:
                corpus.speakers[speaker_key] = Speaker(
                    owner=corpus, id=u[KeySpeaker], meta=speakers_data[u[KeySpeaker]][KeyMeta]
                )
            else:
                corpus.speakers[speaker_key] = Speaker(
                    owner=corpus, id=u[KeySpeaker], meta=speakers_data[u[KeySpeaker]]
                )

        speaker = corpus.speakers[speaker_key]
        speaker.vectors = speakers_data[u[KeySpeaker]].get(KeyVectors, [])

        # temp fix for reddit reply_to
        if "reply_to" in u:
            reply_to_data = u["reply_to"]
        else:
            reply_to_data = u[KeyReplyTo]
        utt = Utterance(
            owner=corpus,
            id=u[KeyId],
            speaker=speaker,
            conversation_id=u[KeyConvoId],
            reply_to=reply_to_data,
            timestamp=u[KeyTimestamp],
            text=u[KeyText],
            meta=u[KeyMeta],
        )
        utt.vectors = u.get(KeyVectors, [])
        corpus.utterances[utt.id] = utt


def merge_utterance_lines(utt_dict):
    """
    For merging adjacent utterances by the same speaker
    """
    new_utterances = {}
    merged_with = {}
    for uid, utt in utt_dict.items():
        merged = False
        if utt.reply_to is not None and utt.speaker is not None:
            u0 = utt_dict[utt.reply_to]
            if u0.conversation_id == utt.conversation_id and u0.speaker == utt.speaker:
                merge_target = merged_with[u0.id] if u0.id in merged_with else u0.id
                new_utterances[merge_target].text += " " + utt.text
                merged_with[utt.id] = merge_target
                merged = True
        if not merged:
            if utt.reply_to in merged_with:
                utt.reply_to = merged_with[utt.reply_to]
            new_utterances[utt.id] = utt
    return new_utterances


def initialize_conversations(corpus, utt_dict, convos_data, convo_to_utts=None):
    """
    Initialize Conversation objects from utterances and conversations data.
    If a mapping from Conversation IDs to their constituent Utterance IDs is
    already known (e.g., as a side effect of a prior computation) they can be
    directly provided via the convo_to_utts parameter, otherwise the mapping
    will be computed by iteration over the Utterances in utt_dict.
    """
    # organize utterances by conversation
    if convo_to_utts is None:
        convo_to_utts = defaultdict(list)  # temp container identifying utterances by conversation
        for u in utt_dict.values():
            convo_key = (
                u.conversation_id
            )  # each conversation_id is considered a separate conversation
            convo_to_utts[convo_key].append(u.id)
    conversations = {}
    for convo_id in convo_to_utts:
        # look up the metadata associated with this conversation, if any
        convo_data = convos_data.get(convo_id, None)
        if convo_data is not None:
            if KeyMeta in convo_data:
                convo_meta = convo_data[KeyMeta]
            else:
                convo_meta = convo_data
        else:
            convo_meta = None

        convo = Conversation(
            owner=corpus, id=convo_id, utterances=convo_to_utts[convo_id], meta=convo_meta
        )

        if convo_data is not None and KeyVectors in convo_data and KeyMeta in convo_data:
            convo.vectors = convo_data.get(KeyVectors, [])
        conversations[convo_id] = convo
    return conversations


def dump_helper_bin(d: ConvoKitMeta, d_bin: Dict, fields_to_skip=None) -> Dict:  # object_idx
    """

    :param d: The ConvoKitMeta to encode
    :param d_bin: The dict of accumulated lists of binary attribs
    :return:
    """
    if fields_to_skip is None:
        fields_to_skip = []

    obj_idx = d.index.get_index(d.obj_type)
    d_out = {}
    for k, v in d.items():
        if k in fields_to_skip:
            continue
        try:
            if obj_idx[k][0] == "bin":
                d_out[k] = "{}{}{}".format(BIN_DELIM_L, len(d_bin[k]), BIN_DELIM_R)
                d_bin[k].append(v)
            else:
                d_out[k] = v
        except KeyError:
            # fails silently (object has no such metadata that was indicated in metadata index)
            pass
    return d_out


def dump_corpus_component(
    corpus, dir_name, filename, obj_type, bin_name, exclude_vectors, fields_to_skip
):
    with open(os.path.join(dir_name, filename), "w") as f:
        d_bin = defaultdict(list)
        objs = defaultdict(dict)
        for obj_id in corpus.get_object_ids(obj_type):
            objs[obj_id][KeyMeta] = dump_helper_bin(
                corpus.get_object(obj_type, obj_id).meta, d_bin, fields_to_skip.get(obj_type, [])
            )
            obj_vectors = corpus.get_object(obj_type, obj_id).vectors
            objs[obj_id][KeyVectors] = (
                obj_vectors
                if exclude_vectors is None
                else list(set(obj_vectors) - set(exclude_vectors))
            )
        json.dump(objs, f)

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-{}-bin.p".format(bin_name)), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)


def dump_utterances(corpus, dir_name, exclude_vectors, fields_to_skip):
    with open(os.path.join(dir_name, "utterances.jsonl"), "w") as f:
        d_bin = defaultdict(list)

        for ut in corpus.iter_utterances():
            ut_obj = {
                KeyId: ut.id,
                KeyConvoId: ut.conversation_id,
                KeyText: ut.text,
                KeySpeaker: ut.speaker.id,
                KeyMeta: dump_helper_bin(ut.meta, d_bin, fields_to_skip.get("utterance", [])),
                KeyReplyTo: ut.reply_to,
                KeyTimestamp: ut.timestamp,
                KeyVectors: ut.vectors
                if exclude_vectors is None
                else list(set(ut.vectors) - set(exclude_vectors)),
            }
            json.dump(ut_obj, f)
            f.write("\n")

        for name, l_bin in d_bin.items():
            with open(os.path.join(dir_name, name + "-bin.p"), "wb") as f_pk:
                pickle.dump(l_bin, f_pk)


def load_jsonlist_to_dict(filename, index_key="id", value_key="value"):
    entries = {}
    with open(filename, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries[entry[index_key]] = entry[value_key]
    return entries


def dump_jsonlist_from_dict(entries, filename, index_key="id", value_key="value"):
    with open(filename, "w") as f:
        for k, v in entries.items():
            json.dump({index_key: k, value_key: v}, f)
            f.write("\n")


def extract_meta_from_df(df):
    meta_cols = [col.split(".")[1] for col in df if col.startswith("meta")]
    return meta_cols


def load_binary_metadata(filename, index, exclude_meta=None):
    binary_data = {"utterance": {}, "conversation": {}, "speaker": {}, "corpus": {}}
    for component_type in binary_data:
        meta_index = index.get_index(component_type)
        for meta_key, meta_type in meta_index.items():
            if meta_type == ["bin"] and (
                exclude_meta is None or meta_key not in exclude_meta[component_type]
            ):
                try:
                    with open(
                        os.path.join(filename, meta_key + "-{}-bin.p".format(component_type)), "rb"
                    ) as f:
                        l_bin = pickle.load(f)
                        binary_data[component_type][meta_key] = l_bin
                except FileNotFoundError:
                    warn(
                        f"Metadata field {meta_key} is specified to have binary type but no saved binary data was found. This field will be skipped."
                    )
    return binary_data


def load_jsonlist_to_db(
    filename,
    db,
    collection_prefix,
    start_line=None,
    end_line=None,
    exclude_meta=None,
    bin_meta=None,
):
    """
    Populate the specified MongoDB database with the utterance data contained in
    the given filename (which should point to an utterances.jsonl file).
    """
    utt_collection = db[f"{collection_prefix}_utterance"]
    meta_collection = db[f"{collection_prefix}_meta"]
    inserted_ids = set()
    speaker_key = None
    convo_key = None
    reply_key = None
    with open(filename) as f:
        utt_insertion_buffer = []
        meta_insertion_buffer = []
        for ln, line in enumerate(f):
            if start_line is not None and ln < start_line:
                continue
            if end_line is not None and ln > end_line:
                break
            utt_obj = json.loads(line)
            if speaker_key is None:
                # backwards compatibility for corpora made before the user->speaker rename
                speaker_key = "speaker" if "speaker" in utt_obj else "user"
            if convo_key is None:
                # backwards compatibility for corpora made before the root->conversation_id rename
                convo_key = "conversation_id" if "conversation_id" in utt_obj else "root"
            if reply_key is None:
                # fix for misnamed reply_to in subreddit corpora
                reply_key = "reply-to" if "reply-to" in utt_obj else "reply_to"
            utt_insertion_buffer.append(
                UpdateOne(
                    {"_id": utt_obj["id"]},
                    {
                        "$set": {
                            "speaker_id": utt_obj[speaker_key],
                            "conversation_id": utt_obj[convo_key],
                            "reply_to": utt_obj[reply_key],
                            "timestamp": utt_obj["timestamp"],
                            "text": utt_obj["text"],
                        }
                    },
                    upsert=True,
                )
            )
            utt_meta = utt_obj["meta"]
            if exclude_meta is not None:
                for exclude_key in exclude_meta:
                    if exclude_key in utt_meta:
                        del utt_meta[exclude_key]
            if bin_meta is not None:
                for key, bin_list in bin_meta.items():
                    bin_locator = utt_meta[key]
                    if (
                        type(bin_locator) == "str"
                        and bin_locator.startswith(BIN_DELIM_L)
                        and bin_locator.endswith(BIN_DELIM_R)
                    ):
                        bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                        utt_meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
            meta_insertion_buffer.append(
                UpdateOne({"_id": "utterance_" + utt_obj["id"]}, {"$set": utt_meta}, upsert=True)
            )
            inserted_ids.add(utt_obj["id"])
            if len(utt_insertion_buffer) >= JSONLIST_BUFFER_SIZE:
                utt_collection.bulk_write(utt_insertion_buffer)
                meta_collection.bulk_write(meta_insertion_buffer)
                utt_insertion_buffer = []
                meta_insertion_buffer = []
        # after loop termination, insert any remaining items in the buffer
        if len(utt_insertion_buffer) > 0:
            utt_collection.bulk_write(utt_insertion_buffer)
            meta_collection.bulk_write(meta_insertion_buffer)
            utt_insertion_buffer = []
            meta_insertion_buffer = []
    return inserted_ids


def load_json_to_db(
    filename, db, collection_prefix, component_type, exclude_meta=None, bin_meta=None
):
    """
    Populate the specified MongoDB database with corpus component data from
    either the speakers.json or conversations.json file located in a directory
    containing valid ConvoKit Corpus data. The component_type parameter controls
    which JSON file gets used.
    """
    component_collection = db[f"{collection_prefix}_{component_type}"]
    meta_collection = db[f"{collection_prefix}_meta"]
    if component_type == "speaker":
        json_data = load_speakers_data_from_dir(filename, exclude_meta)
    elif component_type == "conversation":
        json_data = load_convos_data_from_dir(filename, exclude_meta)
    component_insertion_buffer = []
    meta_insertion_buffer = []
    for component_id, component_data in json_data.items():
        if KeyMeta in component_data:
            # contains non-metadata entries
            payload = {k: v for k, v in component_data.items() if k not in {"meta", "vectors"}}
            meta = component_data[KeyMeta]
        else:
            # contains only metadata, with metadata at the top level
            payload = {}
            meta = component_data
        component_insertion_buffer.append(
            UpdateOne({"_id": component_id}, {"$set": payload}, upsert=True)
        )
        if bin_meta is not None:
            for key, bin_list in bin_meta.items():
                bin_locator = meta[key]
                if (
                    type(bin_locator) == "str"
                    and bin_locator.startswith(BIN_DELIM_L)
                    and bin_locator.endswith(BIN_DELIM_R)
                ):
                    bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                    meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
        meta_insertion_buffer.append(
            UpdateOne({"_id": f"{component_type}_{component_id}"}, {"$set": meta}, upsert=True)
        )
    component_collection.bulk_write(component_insertion_buffer)
    meta_collection.bulk_write(meta_insertion_buffer)


def load_corpus_info_to_db(filename, db, collection_prefix, exclude_meta=None, bin_meta=None):
    """
    Populate the specified MongoDB database with Corpus metadata loaded from the
    corpus.json file of a directory containing valid ConvoKit Corpus data.
    """
    if exclude_meta is None:
        exclude_meta = {}
    meta_collection = db[f"{collection_prefix}_meta"]
    with open(os.path.join(filename, "corpus.json")) as f:
        corpus_meta = {k: v for k, v in json.load(f).items() if k not in exclude_meta}
        if bin_meta is not None:
            for key, bin_list in bin_meta.items():
                bin_locator = corpus_meta[key]
                if (
                    type(bin_locator) == "str"
                    and bin_locator.startswith(BIN_DELIM_L)
                    and bin_locator.endswith(BIN_DELIM_R)
                ):
                    bin_idx = int(bin_locator[len(BIN_DELIM_L) : -len(BIN_DELIM_R)])
                    corpus_meta[key] = bson.Binary(pickle.dumps(bin_list[bin_idx]))
        meta_collection.update_one(
            {"_id": f"corpus_{collection_prefix}"}, {"$set": corpus_meta}, upsert=True
        )


def populate_db_from_file(
    filename,
    db,
    collection_prefix,
    meta_index,
    utterance_start_index,
    utterance_end_index,
    exclude_utterance_meta,
    exclude_conversation_meta,
    exclude_speaker_meta,
    exclude_overall_meta,
):
    """
    Populate all necessary collections of a MongoDB database so that it can be
    used by a DBStorageManager, sourcing data from the valid ConvoKit Corpus
    data pointed to by the filename parameter.
    """
    binary_meta = load_binary_metadata(
        filename,
        meta_index,
        {
            "utterance": exclude_utterance_meta,
            "conversation": exclude_conversation_meta,
            "speaker": exclude_speaker_meta,
            "corpus": exclude_overall_meta,
        },
    )

    # first load the utterance data
    inserted_utt_ids = load_jsonlist_to_db(
        os.path.join(filename, "utterances.jsonl"),
        db,
        collection_prefix,
        utterance_start_index,
        utterance_end_index,
        exclude_utterance_meta,
        binary_meta["utterance"],
    )
    # next load the speaker and conversation data
    for component_type in ["speaker", "conversation"]:
        load_json_to_db(
            filename,
            db,
            collection_prefix,
            component_type,
            (exclude_speaker_meta if component_type == "speaker" else exclude_conversation_meta),
            binary_meta[component_type],
        )
    # finally, load the corpus metadata
    load_corpus_info_to_db(
        filename, db, collection_prefix, exclude_overall_meta, binary_meta["corpus"]
    )

    return inserted_utt_ids


def init_corpus_from_storage_manager(corpus, utt_ids=None):
    """
    Use an already-populated MongoDB database to initialize the components of
    the specified Corpus (which should be empty before this function is called)
    """
    # we will bypass the initialization step when constructing components since
    # we know their necessary data already exists within the db
    corpus.storage.bypass_init = True

    # fetch object ids from the DB and initialize corpus components for them
    # create speakers first so we can refer to them when initializing utterances
    speakers = {}
    for speaker_doc in corpus.storage.data["speaker"].find(projection=["_id"]):
        speaker_id = speaker_doc["_id"]
        speakers[speaker_id] = Speaker(owner=corpus, id=speaker_id)
    corpus.speakers = speakers

    # next, create utterances
    utterances = {}
    convo_to_utts = defaultdict(list)
    for utt_doc in corpus.storage.data["utterance"].find(
        projection=["_id", "speaker_id", "conversation_id"]
    ):
        utt_id = utt_doc["_id"]
        if utt_ids is None or utt_id in utt_ids:
            convo_to_utts[utt_doc["conversation_id"]].append(utt_id)
            utterances[utt_id] = Utterance(
                owner=corpus, id=utt_id, speaker=speakers[utt_doc["speaker_id"]]
            )
    corpus.utterances = utterances

    # run post-construction integrity steps as in regular constructor
    corpus.conversations = initialize_conversations(corpus, corpus.utterances, {}, convo_to_utts)
    corpus.meta_index.enable_type_check()
    corpus.update_speakers_data()

    # restore the StorageManager's init behavior to default
    corpus.storage.bypass_init = False
