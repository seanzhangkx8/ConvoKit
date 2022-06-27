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
            "utterance": None,
            "conversation": None,
            "speaker": None,
            "meta": None
        }

    def get_collection(self, component_type: str):
        if component_type not in self.data:
            raise ValueError('component_type must be one of "utterance", "conversation", "speaker", or "meta".')
        return self.data[component_type]

class MemStorageManager(StorageManager):
    """
    Concrete StorageManager implementation for in-memory data storage.
    Collections are implemented as vanilla Python dicts.
    """
    def __init__(
        self, 
        corpus_id: Optional[str] = None, 
        version: Optional[str] = "0"
    ):
        super().__init__(corpus_id, version)

        # initialize component collections as dicts
        for key in self.data:
            self.data[key] = {}

    def get_data(self, component_type: str, component_id: str, property_name: str):
        """
        Retrieve the property data for the component of type component_type with
        id component_id
        """
        collection = self.get_collection(component_type)
        if component_id not in collection:
            raise KeyError(f"This StorageManager does not have an entry for the {component_type} with id {component_id}.")
        return collection[component_id][property_name]

    def update_data(self, component_type: str, component_id: str, property_name: str, new_value):
        """
        Set or update the property data for the component of type component_type
        with id component_id
        """
        collection = self.get_collection(component_type)
        # don't create new collections if the ID is not found; this is supposed to be handled in the
        # CorpusComponent constructor so if the ID is missing that indicates something is wrong
        if component_id not in collection:
            raise KeyError(f"This StorageManager does not have an entry for the {component_type} with id {component_id}.")
        collection[component_id][property_name] = new_value
        