from typing import Dict, List, Collection, Callable, Set, Generator, Tuple, Optional, ValuesView
from .user import User
from .corpusObject import CorpusObject


class Utterance(CorpusObject):
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance.
    :param user: the user giving the utterance.
    :param root: the id of the root utterance of the conversation.
    :param reply_to: id of the utterance this was a reply to.
    :param timestamp: timestamp of the utterance. Can be any
        comparable type.
    :param text: text of the utterance.
    :type text: str

    :ivar id: the unique id of the utterance.
    :ivar user: the user giving the utterance.
    :ivar root: the id of the root utterance of the conversation.
    :ivar reply_to: id of the utterance this was a reply to.
    :ivar timestamp: timestamp of the utterance.
    :ivar text: text of the utterance.
    """

    def __init__(self, owner=None, id: Optional[str] = None, user: Optional[User] = None,
                 root: Optional[str] = None, reply_to: Optional[str] = None,
                 timestamp: Optional[int] = None, text: Optional[str] = None,
                 meta: Optional[Dict] = None):
        super().__init__(obj_type="utterance", owner=owner, id=id, meta=meta)
        self.user = user
        self.root = root
        self.reply_to = reply_to
        self.timestamp = int(timestamp) if timestamp is not None else timestamp
        self.text = text

    def get(self, key: str):
        if key == "id":
            return self.id
        elif key == "user":
            return self.user
        elif key == "root":
            return self.root
        elif key == "reply_to":
            return self.reply_to
        elif key == "timestamp":
            return self.timestamp
        elif key == "text":
            return self.text
        elif key == "meta":
            return self.meta

    # def copy(self):
    #     """
    #     :return: A duplicate of this Utterance with the same data and metadata
    #     """
    #     return Utterance(id=self.id,
    #                      user=self.user,
    #                      root=self.root,
    #                      reply_to=self.reply_to,
    #                      timestamp=self.timestamp,
    #                      text=self.text,
    #                      meta=self.meta.copy())


    def get_info(self, key):
        """
            Gets attribute <key> of the utterance. Returns None if the user does not have this attribute.
            
            :param key: name of attribute
            :return: attribute <key>
        """
        
        return self.meta.get(key, None)

    def set_info(self, key, value):
        """
            Sets attribute <key> of the utterance to <value>.

            :param key: name of attribute
            :param value: value to set
            :return: None
        """
        self.meta[key] = value

    def __eq__(self, other):
        if not isinstance(other, Utterance):
            return False
        try:
            return self.id == other.id and self.root == other.root and self.reply_to == other.reply_to and \
                   self.user == other.user and self.timestamp == other.timestamp and self.text == other.text
        except AttributeError: # for backwards compatibility with wikiconv
            return self.__dict__ == other.__dict__

    def __str__(self):
        return "Utterance('id': {}, 'root': {}, 'reply-to': {}, " \
               "'user': {}, 'timestamp': {}, 'text': {}, 'meta': {})".format(repr(self.id),
                                                                             self.root,
                                                                             self.reply_to,
                                                                             self.user,
                                                                             self.timestamp,
                                                                             repr(self.text),
                                                                             self.meta)


