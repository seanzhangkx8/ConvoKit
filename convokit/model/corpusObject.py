from warnings import warn
from abc import ABC
import json
from .convoKitIndex import ConvoKitIndex
from .convoKitMeta import ConvoKitMeta


class CorpusObject(ABC):

    def __init__(self, obj_type: str, owner=None, id=None, meta=None):
        self.obj_type = obj_type # utterance, user, conversation
        self._owner = owner
        self.meta = self.init_meta(meta)
        self._id = id

    def get_owner(self):
        return self._owner

    def set_owner(self, owner):
        self._owner = owner
        if owner is not None:
            self.meta = self.init_meta(self.meta)

    owner = property(get_owner, set_owner)

    def init_meta(self, meta):
        if self._owner is None:
            return meta
        elif meta is None:
            ck_meta = ConvoKitMeta(self.owner.meta_index, self.obj_type)
            return ck_meta
        else:
            ck_meta = ConvoKitMeta(self.owner.meta_index, self.obj_type)
            for key, value in meta.items():
                ck_meta[key] = value
            return ck_meta

    def get_id(self):
        return self._id

    def set_id(self, value):
        if not isinstance(value, str):
            self._id = str(value)
            warn("Corpus object id must be a string. Input has been casted to string.")
        self._id = value

    id = property(get_id, set_id)

    def __eq__(self, other):
        if type(self) != type(other): return False
        # do not compare 'utterances' and 'conversations' in User.__dict__. recursion loop will occur.
        self_keys = set(self.__dict__).difference(['_owner', 'utterances', 'conversations'])
        other_keys = set(other.__dict__).difference(['_owner', 'utterances', 'conversations'])
        return self_keys == other_keys and all([self.__dict__[k] == other.__dict__[k] for k in self_keys])
