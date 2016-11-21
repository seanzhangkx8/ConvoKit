"""The objects used to represent a dataset."""

import json
from collections import defaultdict

class User:
    """Represents a single user in a dataset.
   
    :param name: name of the user.
    :type name: str
    :param info: arbitrary dictionary of attributes associated
        with the user.
    :type info: dict

    :ivar name: name of the user.
    :ivar info: dictionary of attributes associated with the user.
    """
    
    def __init__(self, name=None, info={}):
        self._name = name
        self._info = info
        self._update_uid()

    def _get_name(self): return self._name
    def _set_name(self, value):
        self._name = value
        self._update_uid()
    name = property(_get_name, _set_name)

    def _get_info(self): return self._info
    def _set_info(self, value):
        self._info = value
        self._update_uid()
    info = property(_get_info, _set_info)

    def _update_uid(self):
        self._uid = "User({name: '" + self._name + \
                "', info: " + str(self._info) + "})"

    def __eq__(self, other):
        return self._uid == other._uid

    def __hash__(self):
        return hash(self._uid)

    def __repr__(self):
        return self._uid

class Utterance:
    """Represents a single utterance in the dataset.

    :param id: the unique id of the utterance. Can be any hashable type.
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

    def __init__(self, id=None, user=None, root=None, reply_to=None,
            timestamp=None, text=None):
        self.id = id
        self.user = user
        self.root = root
        self.reply_to = reply_to
        self.timestamp = timestamp
        self.text = text

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "Utterance(" + str(self.__dict__) + ")"

class Model:
    """Represents a dataset, which can be loaded from a JSON file or a list
    of utterances.

    :param filename: path of json file to load
    :param utterances: list of utterances to load
    :param merge_lines: whether to merge adjacent
        lines from same author if the two utterances have the same root.
        Uses the older version of the other attribs.

    :ivar utterances: dictionary of utterances in the dataset, indexed by id.
    """

    def __init__(self, filename=None, utterances=None, merge_lines=False):
        KeyId = "id"
        KeyUser = "user"
        KeyConvoRoot = "root"
        KeyReplyTo = "reply-to"
        KeyTimestamp = "timestamp"
        KeyText = "text"
        KeyUserInfo = "user-info"  # can store any extra data

        if filename is not None:
            utterances = json.load(open(filename, "r"))
            self.utterances = {}
            self.all_users = set()
            for u in utterances:
                u = defaultdict(lambda: None, u)
                user = User(name=u[KeyUser], info=u[KeyUserInfo])
                self.all_users.add(user)
                ut = Utterance(id=u[KeyId], user=user,
                        root=u[KeyConvoRoot],
                        reply_to=u[KeyReplyTo], timestamp=u[KeyTimestamp],
                        text=u[KeyText])
                self.utterances[ut.id] = ut
        elif utterances is not None:
            self.utterances = { u.id: u for u in utterances }

        if merge_lines:
            new_utterances = {}
            for uid, u in self.utterances.items():
                merged = False
                if u.reply_to is not None and u.user is not None:
                    u0 = self.utterances[u.reply_to]
                    if u0.root == u.root and u0.user == u.user:
                        new_utterances[u0.id].text += " " + u.text
                        merged = True
                if not merged:
                    new_utterances[u.id] = u
            self.utterances = new_utterances

    def users(self, selector=None):
        """Get users in the dataset.

        :param selector: optional function that takes in a
            `User` and returns True to include the user in the
            resulting list, or False otherwise.

        :return: Set containing all users selected by the selector function,
            or all users in the dataset if no selector function was
            used.
        """
        if selector is None:
            return self.all_users
        else:
            return set([u for u in self.all_users if selector(u)])

    def user_names(self, selector=None):
        """Get names of users in the dataset.

        :param selector: optional function that takes in a
            `User` and returns True to include the user's name in the
            resulting list, or False otherwise.

        :return: Set containing all user names selected by the selector
            function, or all user names in the dataset if no selector function
            was used.
        """
        return set([u.name for u in self.users(selector)])

    def speaking_pairs(self, selector=None, user_names_only=False):
        """Get all directed speaking pairs (a, b) of users such that a replies
            to b at least once in the dataset.

        :param selector: optional function that takes in
            a speaker user and a replied-to user and returns True to include
            the pair in the result, or False otherwise.
        :param user_names_only: if True, return just pairs of
            user names rather than user objects.
        :type user_names_only: bool

        :return: Set containing all speaking pairs selected by the selector
            function, or all speaking pairs in the dataset if no selector
            function was used.
        """
        pairs = set()
        for u2 in self.utterances.values():
            if u2.user is not None and u2.reply_to is not None:
                u1 = self.utterances[u2.reply_to]
                if u1.user is not None:
                    if selector is None or selector(u1.user, u2.user):
                        pairs.add((u2.user.name, u1.user.name) if 
                                user_names_only else (u2.user, u1.user))
        return pairs

    def pairwise_exchanges(self, selector=None, user_names_only=False):
        """Get all directed pairwise exchanges in the dataset.

        :param selector: optional function that takes in a
            speaker user and a replied-to user and returns True to include
            the pair in the result, or False otherwise.
        :param user_names_only: if True, index conversations
            by user names rather than user objects.
        :type user_names_only: bool

        :return: Dictionary mapping (speaker, target) tuples to a list of
            utterances given by the speaker in reply to the target.
        """
        pairs = defaultdict(list)
        for u2 in self.utterances.values():
            if u2.user is not None and u2.reply_to is not None:
                u1 = self.utterances[u2.reply_to]
                if u1.user is not None:
                    if selector is None or selector(u1.user, u2.user):
                        key = ((u2.user.name, u1.user.name) if 
                                user_names_only else (u2.user, u1.user))
                        pairs[key].append(u2)
        return pairs
