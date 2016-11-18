import json
from collections import defaultdict

KeyId = "id"
KeyUser = "user"
KeyConvoRoot = "root"
KeyReplyTo = "reply-to"
KeyTimestamp = "timestamp"
KeyText = "text"
KeyUserInfo = "user-info"  # can store any extra data

class User:
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
    # filename: path of json file to load
    # utterances: list of utterances to load
    # merge_lines: whether to merge adjacent lines from same author if
    #     the two utterances have the same root. Uses the older version
    #     of the other attribs.
    def __init__(self, filename=None, utterances=None, merge_lines=False):
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

    # selector: optional function that takes in an utterance and returns
    #     a bool of whether to include the user of the utterance
    def users(self, selector=None):
        if selector is None:
            return self.all_users
        else:
            return set([u for u in self.all_users if selector(u)])

    def user_names(self, selector=None):
        return set([u.name for u in self.users(selector)])

    def speaking_pairs(self):
        pairs = set()
        for u2 in self.utterances.values():
            if u2.user is not None and u2.reply_to is not None:
                u1 = self.utterances[u2.reply_to]
                if u1.user is not None:
                    pairs.add((u2.user, u1.user))
        return pairs

    def pairwise_convos(self):
        pairs = defaultdict(list)
        for u2 in self.utterances.values():
            if u2.user is not None and u2.reply_to is not None:
                u1 = self.utterances[u2.reply_to]
                if u1.user is not None:
                    pairs[u2.user, u1.user].append(u2)
        return pairs
