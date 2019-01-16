"""The objects used to represent a dataset."""

import json
from functools import total_ordering, reduce
from collections import defaultdict
import os

pair_delim = '-q-a-'

@total_ordering
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
        self._split_attribs = set()
        self._update_uid()

    def identify_by_attribs(self, attribs):
        """Identify a user by a list of attributes. Sets which user info
        attributes should distinguish users of the same name in equality tests.
        For example, in the Supreme Court dataset, users are labeled with the
        current case id. Call this method with attribs = ["case"] to count
        the same person across different cases as different users.

        :param attribs: Collection of attribute names.
        :type attribs: Collection
        """

        self._split_attribs = set(attribs)
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
    meta = property(_get_info, _set_info)

    def _update_uid(self):
        rep = {}
        rep["name"] = self._name
        if self._split_attribs:
            rep["attribs"] = {k: self._info[k] for k in self._split_attribs
                    if k in self._info}
        self._uid = "User(" + str(sorted(rep.items())) + ")"

    def __eq__(self, other):
        return self._uid == other._uid

    def __lt__(self, other):
        return self._uid < other._uid

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
            timestamp=None, text=None, other=None, meta=None):
        self.id = id
        self.user = user
        self.root = root
        self.reply_to = reply_to
        self.timestamp = timestamp
        self.text = text
        self.other = other
        self.meta = meta if meta is not None else {}

    def get(self, key):
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
            return text

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "Utterance(" + str(self.__dict__) + ")"

class Conversation:
    """Represents a discrete subset of utterances in the dataset, connected by a
    reply-to chain.
    
    :param owner: The Corpus that this Conversation belongs to
    :param id: The unique ID of this Conversation
    :param utterances: A list of the IDs of the Utterances in this Conversation
    :param meta: Table of initial values for conversation-level metadata

    :ivar meta: A dictionary-like view object providing read-write access to
        conversation-level metadata. For utterance-level metadata, use 
        Utterance.meta. For user-level metadata, use User.meta. For corpus-level
        metadata, use Corpus.meta.
    """

    def __init__(self, owner, id=None, utterances=None, meta=None):
        self._owner = owner
        self._id = id
        self._utterance_ids = utterances
        self._usernames = None
        self._meta = {} if meta is None else meta

    # Conversation.meta property
    def _get_meta(self):
        """Provides read-write access to conversation-level metadata. For
        utterance-level metadata, use Utterance.meta. For user-level metadata,
        use User.meta. For corpus-level metadata, use Corpus.meta."""
        return self._meta
    def _set_meta(self, new_meta):
        self._meta = new_meta
    meta = property(_get_meta, _set_meta)

    # Conversation.id property
    def _get_id(self):
        """The unique ID of this Conversation [read-only]"""
        return self._id
    id = property(_get_id)

    def get_utterance_ids(self):
        """Produces a list of the unique IDs of all utterances in the 
        Conversation, which can be used in calls to get_utterance() to retrieve 
        specific utterances. Provides no ordering guarantees for the list.
        
        :return: a list of IDs of Utterances in the Conversation
        """
        # we construct a new list instead of returning self._utterance_ids in
        # order to prevent the user from accidentally modifying the internal
        # ID list (since lists are mutable)
        return [ut_id for ut_id in self._utterance_ids]

    def get_utterance(self, ut_id):
        """Looks up the Utterance associated with the given ID. Raises a 
        KeyError if no utterance by that ID exists.
        
        :return: the Utterance with the given ID
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_utterance(ut_id)

    def iter_utterances(self):
        """Generator allowing iteration over all utterances in the Conversation. 
        Provides no ordering guarantees.
        
        :return: Generator that produces Users
        """
        for ut_id in self._utterance_ids:
            yield self._owner.get_utterance(ut_id)

    def get_usernames(self):
        """Produces a list of names of all users in the Conversation, which can 
        be used in calls to get_user() to retrieve specific users. Provides no 
        ordering guarantees for the list.
        
        :return: a list of usernames
        """
        if self._usernames is None:
            # first call to get_usernames or iter_users; precompute cached list
            # of usernames
            self._usernames = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._usernames.add(ut.user.name)
        return list(self._usernames)

    def get_user(self, username):
        """Looks up the User with the given name. Raises a KeyError if no user 
        with that name exists.

        :return: the User with the given username
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_user(username)

    def iter_users(self):
        """Generator allowing iteration over all users in the Conversation. 
        Provides no ordering guarantees.

        :return: Generator that produces Users.
        """
        if self._usernames is None:
            # first call to get_usernames or iter_users; precompute cached list
            # of usernames
            self._usernames = set()
            for ut_id in self._utterance_ids:
                ut = self._owner.get_utterance(ut_id)
                self._usernames.add(ut.user.name)
        for username in self._usernames:
            yield self._owner.get_user(username)

KeyId = "id"
KeyUser = "user"
KeyConvoRoot = "root"
KeyReplyTo = "reply-to"
KeyTimestamp = "timestamp"
KeyText = "text"
DefinedKeys = set([KeyId, KeyUser, KeyConvoRoot, KeyReplyTo,
    KeyTimestamp, KeyText])
KeyMeta = "meta"

class Corpus:
    """Represents a dataset, which can be loaded from a JSON file, CSV file or a
    list of utterances.

    If a CSV file, the first row should consist of key names. Unknown key names
    will have their attributes added to the "user-info" entry.

    :param filename: path of json or csv file to load
    :param utterances: list of utterances to load
    :param merge_lines: whether to merge adjacent
        lines from the same user if the two utterances have the same root.
    :param subdivide_users_by: collection of strings corresponding to attribute
        names defined in the "user-info" entry. Use this if you want to count
        the same user as being different depending on attributes other than
        username. For example, in the Supreme Court dataset, users are annotated
        with the current case id. You could use this to count the same person
        across different cases as being different users.
    :param delim: if loading a csv, specifies the delimiter string.

    :ivar utterances: dictionary of utterances in the dataset, indexed by id.
    """

    def __init__(self, filename=None, utterances=None, merge_lines=False,
        subdivide_users_by=[], delim=","):
        self.meta = {}

        if filename is not None:
            if os.path.isdir(filename):
                with open(os.path.join(filename, "utterances.json"), "r") as f:
                    utterances = json.load(f)
                with open(os.path.join(filename, "users.json"), "r") as f:
                    users_meta = defaultdict(dict)
                    for k, v in json.load(f).items():
                        users_meta[k] = v
                with open(os.path.join(filename, "conversations.json"), "r") as f:
                    convos_meta = defaultdict(dict)
                    for k, v in json.load(f).items():
                        convos_meta[k] = v
                with open(os.path.join(filename, "corpus.json"), "r") as f:
                    self.meta = json.load(f)
            else:
                users_meta = defaultdict(dict)
                convos_meta = defaultdict(dict)
                with open(filename, "r") as f:
                    try:
                        utterances = json.load(f)
                    except:
                        try:
                            utterances = self._load_csv(f, delim, DefinedKeys)
                        except:
                            raise ValueError("Couldn't load corpus:" +
                                " unknown file type")

                #with open(".".join(filename.split(".")[:-1]) + "-users.json", "r") as f:
                #    for k, v in json.load(f).items():
                #        users_meta[k] = v

            self.utterances = {}
            self.all_users = set()
            users_cache = {}   # avoids creating duplicate user objects
            #print(len(utterances))
            for i, u in enumerate(utterances):
                # print(u)
                #if i % 100000 == 0: print(i, end=" ", flush=True)
                u = defaultdict(lambda: None, u)

                # handle this utterance's user info
                #user_key = (u[KeyUser], str(sorted(u[KeyUserInfo].items())) if
                #    u[KeyUserInfo] is not None else None)
                user_key = u[KeyUser]
                if user_key not in users_cache:
                    users_cache[user_key] = User(name=u[KeyUser],
                        info=users_meta[u[KeyUser]])
                user = users_cache[user_key]
                self.all_users.add(user)

                other_keys = list(u.keys())
                other_keys.remove(KeyText)
                other = {}
                for key in other_keys:
                    other[key] = u[key]
                ut = Utterance(id=u[KeyId], user=user,
                        root=u[KeyConvoRoot],
                        reply_to=u[KeyReplyTo], timestamp=u[KeyTimestamp],
                        text=u[KeyText], other=other, meta=u[KeyMeta])
                self.utterances[ut.id] = ut
        elif utterances is not None:
            self.all_users = set([u.user for u in utterances])
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

        # organize utterances by conversation
        convo_to_utts = defaultdict(list) # temp container identifying utterances by conversation
        for u in self.utterances.values():
            convo_key = u.root # each root is considered a separate conversation
            convo_to_utts[convo_key].append(u.id)
        self.conversations = {}
        for convo_id in convo_to_utts:
            # look up the metadata associated with this conversation, if any
            convo_meta = convos_meta.get(convo_id, None)
            convo = Conversation(self, id=convo_id,
                        utterances=convo_to_utts[convo_id],
                        meta=convo_meta)
            self.conversations[convo_id] = convo

        if subdivide_users_by:
            self.subdivide_users_by(subdivide_users_by)

    def dump(self, dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        with open(os.path.join(dir_name, "corpus.json"), "w") as f:
            json.dump(self.meta, f)
        with open(os.path.join(dir_name, "users.json"), "w") as f:
            users = {u: self.get_user(u).meta for u in self.get_usernames()}
            json.dump(users, f)
        with open(os.path.join(dir_name, "conversations.json"), "w") as f:
            convos = {c: self.get_conversation(c).meta for c in self.get_conversation_ids()}
            json.dump(convos, f)
        with open(os.path.join(dir_name, "utterances.json"), "w") as f:
            uts = []
            for ut in self.iter_utterances():
                uts.append({
                    KeyId: ut.id,
                    KeyConvoRoot: ut.root,
                    KeyText: ut.text,
                    KeyUser: ut.user.name,
                    KeyMeta: ut.meta
                })
            json.dump(uts, f)

    def get_utterance_ids(self):
        return self.utterances.keys()

    def get_utterance(self, ut_id):
        return self.utterances[ut_id]

    def iter_utterances(self):
        for v in self.utterances.values():
            yield v

    def get_conversation_ids(self):
        return self.conversations.keys()

    def get_conversation(self, cid):
        return self.conversations[cid]

    def iter_conversations(self):
        for v in self.conversations.values():
            yield v
            
    def _load_csv(self, f, delim, defined_keys):
        keys = f.readline()[:-1].split(delim)
        utterances = []
        for line in f:
            values = line[:-1].split(delim)
            utterance = {"user-info": {}}
            for k, v in zip(keys, values):
                if k in defined_keys:
                    utterance[k] = v
                else:
                    utterance["user-info"][k] = v
            utterances.append(utterance)
        return utterances

    def subdivide_users_by(self, attribs):
        """Use this if you want to count the same user as being different
        depending on attributes other than username. For example, in the Supreme
        Court dataset, users are annotated with the current case id. You could
        use this to count the same person across different cases as being
        different users.

        Repeated calls to this method will override previous subdivisions.

        :param attribs: Collection of attribute names to subdivide users on.
        :type attribs: Collection
        """

        new_all_users = set()
        for u in self.utterances.values():
            if u.user is not None:
                u.user.identify_by_attribs(attribs)
                new_all_users.add(u.user)
        self.all_users = new_all_users

    def filter_utterances_by(self, regular_kv_pairs={},
        user_info_kv_pairs={}, other_kv_pairs={}):
        """
        Creates a subset of the utterances filtered by certain attributes. Irreversible.
        If the method is run again, it will filter the already filtered subset.
        Always takes the intersection of the specified key-pairs
        """
        new_utterances = {}
        regular_keys = list(regular_kv_pairs.keys())
        user_info_keys = list(user_info_kv_pairs.keys())
        other_keys = list(other_kv_pairs.keys())
        for uid, utterance in self.utterances.items():
            user_info = utterance.user._get_info()
            other_dict = utterance.other
            regular = all(utterance.get(key) == regular_kv_pairs[key] for key in regular_keys)
            user = all(user_info[key] == user_info_kv_pairs[key] for key in user_info_keys)
            other = all(other_dict[key] == other_kv_pairs[key] for key in other_keys)
            if regular and user and other:
                new_utterances[uid] = utterance

        self.utterances = new_utterances

#    def earliest_n_utterances(self, n, uts=None):
#        """Returns the first n utterances (ordered by time)."""
#        if uts is None:
#            uts = self.utterances
#        uts = list(sorted(uts.values(), key=lambda u: u.timestamp))
#        return uts[:n]

    def utterance_threads(self, prefix_len=None, suffix_len=0):
        """
        Returns dict of threads, where a thread is all utterances with the
        same root.

        :param prefix_len: if an integer n, only get the first n utterances
            of each thread (sorted by ascending timestamp value)
        :param suffix_len: if an integer n, only get the last n utterances
            of each thread (sorted by descending timestamp value)

        :return: Dictionary from thread root ids to threads, where a thread is
            itself a dictionary from utterance ids to utterances.
        """
        threads = defaultdict(list)
        for ut in self.utterances.values():
            threads[ut.root].append(ut)
        return {root: {ut.id: ut for ut in list(sorted(l,
            key=lambda ut: ut.timestamp))[-suffix_len:prefix_len]}
            for root, l in threads.items()}

    def get_meta(self):
        return self.meta

    def add_meta(self, key, value):
        self.meta[key] = value

    def iter_users(self, selector=None):
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

    def get_user(self, name):
        return [u for u in self.all_users if u.name == name][0]

    def get_usernames(self, selector=None):
        """Get names of users in the dataset.

        :param selector: optional function that takes in a
            `User` and returns True to include the user's name in the
            resulting list, or False otherwise.

        :return: Set containing all user names selected by the selector
            function, or all user names in the dataset if no selector function
            was used.
        """
        return set([u.name for u in self.iter_users(selector)])

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
            if u2.user is not None and u2.reply_to is not None and \
                u2.reply_to in self.utterances:
                u1 = self.utterances[u2.reply_to]
                if u1.user is not None:
                    if selector is None or selector(u2.user, u1.user):
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
                    if selector is None or selector(u2.user, u1.user):
                        key = ((u2.user.name, u1.user.name) if
                                user_names_only else (u2.user, u1.user))
                        pairs[key].append(u2)
        return pairs

    def iterate_by(self, iter_type, is_utterance_question):
        """Iterator for utterances.

        Can give just questions, just answers or questions followed by their answers
        """
        i = -1
        for utterance in self.utterances.values():
            if utterance.reply_to is not None:
                root_text = self.utterances[utterance.reply_to].text
                if is_utterance_question(root_text):
                    i += 1
                    if iter_type == 'answers':
                        pair_idx = utterance.reply_to + pair_delim + utterance.id
                        yield utterance.id, utterance.text, pair_idx
                        continue
                    question = self.utterances[utterance.reply_to]
                    pair_idx = question.id + pair_delim + utterance.id
                    yield question.id, question.text, pair_idx
                    if iter_type == 'both':
                        pair_idx = utterance.reply_to + pair_delim + utterance.id
                        yield utterance.id, utterance.text, pair_idx
