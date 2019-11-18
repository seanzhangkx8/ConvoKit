from typing import Dict, List, Collection, Callable, Set, Generator, Tuple, Optional, ValuesView
from .utterance import Utterance
from .user import User

class Conversation:
    """Represents a discrete subset of utterances in the dataset, connected by a
    reply-to chain.

    :param owner: The Corpus that this Conversation belongs to
    :param cid: The unique ID of this Conversation
    :param utterances: A list of the IDs of the Utterances in this Conversation
    :param meta: Table of initial values for conversation-level metadata

    :ivar meta: A dictionary-like view object providing read-write access to
        conversation-level metadata. For utterance-level metadata, use
        Utterance.meta. For user-level metadata, use User.meta. For corpus-level
        metadata, use Corpus.meta.
    """

    def __init__(self, owner, id: Optional[str]=None,
                 utterances: Optional[List[str]]=None,
                 meta: Optional[Dict]=None):
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

    def add_meta(self, key: str, value) -> None:
        """
        Adds a key-value pair to the Conversation metadata

        :return: None
        """
        self.meta[key] = value

    def get_info(self, key):
        """
            Gets attribute <key> of the conversation. Returns None if the conversation does not have this attribute.
            
            :param key: name of attribute
            :return: attribute <key>
        """
        
        return self.meta.get(key,None)

    def set_info(self, key, value):
        """
            Sets attribute <key> of the conversation to <value>.

            :param key: name of attribute
            :param value: value to set
            :return: None
        """

        self.meta[key] = value

    # Conversation.id property
    def _get_id(self):
        """The unique ID of this Conversation [read-only]"""
        return self._id
    id = property(_get_id)

    def get_utterance_ids(self) -> List[str]:
        """Produces a list of the unique IDs of all utterances in the
        Conversation, which can be used in calls to get_utterance() to retrieve
        specific utterances. Provides no ordering guarantees for the list.

        :return: a list of IDs of Utterances in the Conversation
        """
        # we construct a new list instead of returning self._utterance_ids in
        # order to prevent the user from accidentally modifying the internal
        # ID list (since lists are mutable)
        return [ut_id for ut_id in self._utterance_ids]

    def get_utterance(self, ut_id: str) -> Utterance:
        """Looks up the Utterance associated with the given ID. Raises a
        KeyError if no utterance by that ID exists.

        :return: the Utterance with the given ID
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_utterance(ut_id)

    def iter_utterances(self) -> Generator[Utterance, None, None]:
        """Generator allowing iteration over all utterances in the Conversation.
        Provides no ordering guarantees.

        :return: Generator that produces Users
        """
        for ut_id in self._utterance_ids:
            yield self._owner.get_utterance(ut_id)

    def get_usernames(self) -> List[str]:
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

    def get_user(self, username: str) -> User:
        """Looks up the User with the given name. Raises a KeyError if no user
        with that name exists.

        :return: the User with the given username
        """
        # delegate to the owner Corpus since Conversation does not itself own
        # any Utterances
        return self._owner.get_user(username)

    def iter_users(self) -> Generator[User, None, None]:
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

    def check_integrity(self, verbose=True):
        if verbose: print("Checking reply-to chain of Conversation", self.id)
        utt_reply_tos = {utt.id: utt.reply_to for utt in self.iter_utterances()}
        target_utts = set(list(utt_reply_tos.values()))
        speaker_utts = set(list(utt_reply_tos.keys()))
        root_utt = target_utts - speaker_utts # There should only be 1 root_utt: None

        if len(root_utt) != 1:
            if verbose:
                for utt_id in root_utt:
                    if utt_id is not None:
                        print("ERROR: Missing utterance", utt_id)
            return False

        # sanity check
        utts_replying_to_none = 0
        for utt in self.iter_utterances():
            if utt.reply_to is None:
                utts_replying_to_none += 1

        if utts_replying_to_none > 1:
            if verbose: print("ERROR: Found more than one Utterance replying to None.")
            return False
        if verbose: print("No issues found.\n")
        return True

    def _print_convo_helper(self, root: str, indent: int, reply_to_dict: Dict[str, str],
                            utt_info_func: Callable[[Utterance], str]):
        print(" "*indent + utt_info_func(self.get_utterance(root)))
        children_utt_ids = [k for k, v in reply_to_dict.items() if v == root]
        for child_utt_id in children_utt_ids:
            self._print_convo_helper(root=child_utt_id, indent=indent+4,
                                     reply_to_dict=reply_to_dict, utt_info_func=utt_info_func)

    def print_conversation_structure(self, utt_info_func: Callable[[Utterance], str] = lambda utt: utt.user.id):
        if not self.check_integrity(verbose=False):
            raise ValueError("Could not print conversation structure: The utterance reply-to chain is broken. "
                             "Try check_integrity() to diagnose the problem.")

        root_utt_id = [utt for utt in self.iter_utterances() if utt.reply_to is None][0].id
        reply_to_dict = {utt.id: utt.reply_to for utt in self.iter_utterances()}

        self._print_convo_helper(root=root_utt_id, indent=0, reply_to_dict=reply_to_dict, utt_info_func=utt_info_func)

    def __eq__(self, other):
        if isinstance(other, Conversation):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        return "Conversation(" + str(self.__dict__) + ")"
