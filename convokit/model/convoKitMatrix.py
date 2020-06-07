import pandas as pd
from typing import Optional, List

class ConvoKitMatrix:
    """
    A ConvoKitMatrix stores the vector representations of some set of Corpus components (i.e. Utterances,
    Conversations, Speakers).
    """

    def __init__(self, name, matrix, ids: List[str] = None, columns: Optional[List[str]] = None):
        self.name = name
        self.matrix = matrix
        self.ids = ids
        self.columns = columns
        self.ids_to_idx = {id: idx for idx, id in enumerate(ids)}
        self.cols_to_idx = {col: idx for idx, col in enumerate(columns)}
        self._initialization_checks()

    def _initialization_checks(self):
        try:
            self.matrix.shape
        except AttributeError:
            raise AttributeError("Input matrix is not a numpy or scipy matrix.")

        try:
            assert len(self.ids) == self.matrix.shape[0]
            if self.columns is not None:
                assert len(self.columns) == self.matrix.shape[1]
        except AssertionError:
            raise ValueError("Input matrix dimensions {} do not match "
                             "length of ids and/or columns".format(self.matrix.shape))

    def get_vector(self, id: str, columns: Optional[List[str]] = None):
        if columns is None:
            return self.matrix[self.ids_to_idx[id]] # TODO compatible with csr?
        else:
            col_indices = [self.cols_to_idx[col] for col in columns]
            return self.matrix[self.ids_to_idx[id]][col_indices]

    def get_vectors(self, ids: List[str], columns: Optional[List[str]] = None):
        indices = [self.ids_to_idx[k] for k in ids]
        if columns is None:
            return self.matrix[indices]
        else:
            col_indices = [self.cols_to_idx[col] for col in columns]
            return self.matrix[indices][col_indices]

    def to_dict(self):
        if self.columns is None:
            raise ValueError("Matrix columns are missing. Update matrix.columns with a list of column names.")
        d = dict()
        for id, idx in self.ids_to_idx.items():
            row = self.matrix[idx]
            d[id] = {self.columns[i]: v for i, v in enumerate(row)}
        return d

    def to_dataframe(self):
        index = {idx: id_ for id_, idx in self.ids_to_idx.items()}
        sorted_ids = [index[idx] for idx in sorted(index)]

        return pd.DataFrame(self.matrix, index=sorted_ids, columns=self.columns) # TODO check if this passes for None

    def __repr__(self):
        # TODO check this. Maybe make it more consistent with usual matrices
        return "ConvoKitMatrix('name': {}, 'matrix': {}, 'columns': {}, 'ids_to_idx': {})".format(self.name,
                                                                                                  self.matrix,
                                                                                                  self.columns,
                                                                                                  self.ids_to_idx)

    def __str__(self):
        return repr(self)
