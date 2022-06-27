from typing import Optional

class StorageManager:
    """
    Abstraction layer for the concrete representation of data and metadata
    within corpus components (e.g., Utterance text and timestamps). All requests
    to access or modify corpusComponent fields (with the exception of ID) are
    actually routed through one of StorageManager's concrete subclasses. Each
    subclass implements a storage backend that contains the actual data.
    """
    def __init__(
        self,
        corpus_id: Optional[str] = None,
        version: Optional[str] = "0"
    ):
        self.corpus_id = corpus_id
        self.version = version

        # concrete data storage (i.e., collections) for each component type
        # this will be assigned in subclasses
        self.data = {
            "utterance": {},
            "conversation": {},
            "speaker": {},
            "meta": {}
        }

    def get_collection(self, component_type: str):
        if component_type not in self.data:
            raise ValueError('component_type must be one of "utterance", "conversation", "speaker", or "meta".')
        return self.data[component_type]

        