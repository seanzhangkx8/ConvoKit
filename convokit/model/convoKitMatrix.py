import pandas as pd
from typing import Optional, List
import pickle
import os
import numpy as np
from convokit.util import warn
from scipy import sparse

class ConvoKitMatrix:
    """
    A ConvoKitMatrix stores the vector representations of some set of Corpus components (i.e. Utterances,
    Conversations, Speakers).

    :param name:
    :param matrix:
    :param ids:
    :param columns:

    :ivar name:
    :ivar matrix:
    :ivar ids:
    :ivar columns:
    :ivar ids_to_idx:
    :ivar cols_to_idx:
    """

    def __init__(self, name, matrix, ids: List[str] = None, columns: Optional[List[str]] = None):
        self.name = name
        self.matrix = matrix
        self.ids = ids
        if columns is None:
            columns = np.arange(matrix.shape[1])
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

    def get_vectors(self, ids: List[str], as_dataframe=False, columns: Optional[List[str]] = None):
        """

        :param ids: object ids to get vectors for
        :param as_dataframe: whether to return the vector as a dataframe (True) or in its raw array form (False). False
            by default.
        :param columns: optional list of named columns of the vector to include. All columns returned otherwise.
        :return:
        """
        if not isinstance(ids, list):
            is_singleton = True
            ids = [ids]
        else:
            is_singleton = False
        indices = [self.ids_to_idx[k] for k in ids]
        if columns is None:
            if not as_dataframe:
                return self.matrix[indices]
            else:
                matrix = self.matrix.toarray() if sparse.issparse(self.matrix) else self.matrix
                return pd.DataFrame(matrix[indices], index=ids, columns=self.columns)
        else:
            col_indices = [self.cols_to_idx[col] for col in columns]
            matrix = self.matrix.toarray() if sparse.issparse(self.matrix) else self.matrix
            submatrix = matrix[indices, col_indices].reshape(len(indices), len(col_indices))
            if as_dataframe:
                return pd.DataFrame(submatrix, index=ids, columns=columns)
            else:
                if is_singleton:
                    return submatrix.flatten()
                else:
                    return submatrix
            # return submatrix if not as_dataframe else pd.DataFrame(submatrix, index=ids, columns=columns)

    def to_dict(self):
        if self.columns is None:
            raise ValueError("Matrix columns are missing. Update matrix.columns with a list of column names.")
        d = dict()
        for id, idx in self.ids_to_idx.items():
            row = self.matrix[idx]
            d[id] = {self.columns[i]: v for i, v in enumerate(row)}
        return d

    def to_dataframe(self):
        """
        Converts the matrix of vectors into a pandas DataFrame.

        :return: a pandas DataFrame
        """
        index = {idx: id_ for id_, idx in self.ids_to_idx.items()}
        sorted_ids = [index[idx] for idx in sorted(index)]
        matrix = self.matrix.toarray() if sparse.issparse(self.matrix) else self.matrix
        return pd.DataFrame(matrix, index=sorted_ids, columns=self.columns)

    @staticmethod
    def from_file(filepath):
        """
        Initialize a ConvoKitMatrix from a file of form "vector.[name].p".

        :param filepath:
        :return:
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def from_dir(dirpath, matrix_name):
        """
        Initialize a ConvoKitMatrix of the specified `matrix_name` from a specified directory `dirpath`.

        :param dirpath: path to Corpus directory
        :param matrix_name: name of vector matrix
        :return: the initialized ConvoKitMatrix
        """
        try:
            with open(os.path.join(dirpath, 'vectors.{}.p'.format(matrix_name)), 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            warn("Could not find vector with name: {} at {}.".format(matrix_name, dirpath))
            return None

    def dump(self, dirpath):
        """
        Dumps the ConvoKitMatrix as a pickle file.

        :param dirpath: directory path to Corpus
        :return: None
        """
        if not sparse.issparse(self.matrix):
            temp = self.matrix
            self.matrix = sparse.csr_matrix(self.matrix)
            with open(os.path.join(dirpath, 'vectors.{}.p'.format(self.name)), 'wb') as f:
                pickle.dump(self, f)
            self.matrix = temp
        else:
            with open(os.path.join(dirpath, 'vectors.{}.p'.format(self.name)), 'wb') as f:
                pickle.dump(self, f)

    def __repr__(self):
        return "ConvoKitMatrix('name': {}, 'matrix': {})".format(self.name, repr(self.matrix))

    def __str__(self):
        return repr(self)
