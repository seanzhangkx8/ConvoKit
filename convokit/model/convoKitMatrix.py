import pandas as pd
from typing import Optional, List
import pickle
import os
import numpy as np
from convokit.util import warn
from scipy.sparse import issparse, csr_matrix, hstack, vstack

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

        indices = [self.ids_to_idx[k] for k in ids]
        if columns is None:
            if not as_dataframe:
                return self.matrix[indices]
            else:
                matrix = self.matrix.toarray() if issparse(self.matrix) else self.matrix
                return pd.DataFrame(matrix[indices], index=ids, columns=self.columns)
        else:
            col_indices = [self.cols_to_idx[col] for col in columns]
            matrix = self.matrix.toarray() if issparse(self.matrix) else self.matrix
            submatrix = matrix[indices, col_indices].reshape(len(indices), len(col_indices))
            if as_dataframe:
                return pd.DataFrame(submatrix, index=ids, columns=columns)
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

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the matrix of vectors into a pandas DataFrame.

        :return: a pandas DataFrame
        """
        index = {idx: id_ for id_, idx in self.ids_to_idx.items()}
        sorted_ids = [index[idx] for idx in sorted(index)]
        matrix = self.matrix.toarray() if issparse(self.matrix) else self.matrix
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
        if not issparse(self.matrix):
            temp = self.matrix
            self.matrix = csr_matrix(self.matrix)
            with open(os.path.join(dirpath, 'vectors.{}.p'.format(self.name)), 'wb') as f:
                pickle.dump(self, f)
            self.matrix = temp
        else:
            with open(os.path.join(dirpath, 'vectors.{}.p'.format(self.name)), 'wb') as f:
                pickle.dump(self, f)

    def subset(self, ids=None, columns=None):
        """
        Get a (subset) copy of the ConvoKitMatrix object according to specified subset of ids and columns
        :param ids: list of ids to be included in the subset; all by default
        :param columns: list of columns to be included in the subset; all by default
        :return: a new ConvoKitMatrix object with the subset of
        """
        ids = ids if ids is not None else self.ids
        columns = columns if columns is not None else self.columns

        submatrix = self.to_dataframe().loc[ids][columns]
        return ConvoKitMatrix(name=self.name,
                              matrix=csr_matrix(submatrix.values.astype('float64')),
                              ids=ids,
                              columns=columns)

    @staticmethod
    def hstack(name, matrices: List['ConvoKitMatrix']):
        """
        Combines multiple ConvoKitMatrices into a single ConvoKitMatrix by stacking them horizontally (i.e. each
        constituent matrix must have the same ids).

        :param name: name of new matrix
        :param matrices: constituent ConvoKiMatrices
        :return: a new ConvoKitMatrix
        """
        assert len(matrices) > 1
        stacked = hstack([csr_matrix(m.matrix) for m in matrices])
        columns = []
        for m in matrices:
            columns.extend(m.columns)

        return ConvoKitMatrix(name=name,
                              matrix=stacked,
                              ids=matrices[0].ids,
                              columns=columns)

    @staticmethod
    def vstack(name: str, matrices: List['ConvoKitMatrix']):
        """
        Combines multiple ConvoKitMatrices into a single ConvoKitMatrix by stacking them horizontally (i.e. each
        constituent matrix must have the same columns).

        :param name: name of new matrix
        :param matrices: constituent ConvoKiMatrices
        :return: a new ConvoKitMatrix
        """
        assert len(matrices) > 1
        stacked = vstack([csr_matrix(m.matrix) for m in matrices])
        ids = []
        for m in matrices:
            ids.extend(list(m.ids))

        return ConvoKitMatrix(name=name,
                              matrix=stacked,
                              ids=ids,
                              columns=matrices[0].columns)

    def __repr__(self):
        return "ConvoKitMatrix('name': {}, 'matrix': {})".format(self.name, repr(self.matrix))

    def __str__(self):
        return repr(self)