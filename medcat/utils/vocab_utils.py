import numpy as np
import logging
from typing import Type

from medcat.cdb import CDB
from medcat.vocab import Vocab


logger = logging.getLogger(__name__)


def calc_matrix(vocab: Vocab, target_size: int) -> np.ndarray:
    """Calculate the transformation matrix based on the word vectors in the Vocab.

    Performs Principal Component Analysis (PCA).
    This first means all the word vectors in the Vocab.
    It then finds the covariance matrix.
    After that, the eigenvalues and and eigenvectors are calculated.
    And the `target_size` eigenvectors corresponding to the largest
    eigenvalues are selected to create the transformation matrix.

    Args:
        vocab (Vocab): The Vocab.
        target_size (int): The target vector size.

    Returns:
        np.ndarray: The transformation matrix.
    """
    all_vecs = np.vstack(
        [value['vec'] for value in vocab.vocab.values() if value['vec'] is not None]
    )
    logger.debug("Vocab vectors have a total shape of %s", np.shape(all_vecs))
    all_vecs_meaned = all_vecs - np.mean(all_vecs, axis=0)
    cov_matrix = np.cov(all_vecs_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    logger.debug("The sorted eigenvalues are as follows:",
                 [f"{v:5.2f}" for v in eigenvalues[sorted_idx]])
    sorted_eigenvectors = eigenvectors[:, sorted_idx]
    transformation_matrix = sorted_eigenvectors[:, :target_size]
    return transformation_matrix.T


def convert_vec(cur: np.ndarray, matrix: np.ndarray,
                target_dtype: Type = np.float32) -> np.ndarray:
    """Helper function to convert the vector.

    This also guarantees uniform typing (of np.float32) since in our
    experience some vectors may be of a different type before (i.e np.float64).

    Args:
        cur (np.ndarray): The current vector.
        matrix (np.ndarray): The transformation matrix.
        target_dtype (Type): The target element data ype. Defaults to np.float32.

    Returns:
        np.ndarray: The transformed vector.
    """
    return (matrix @ cur).astype(target_dtype)


def convert_vocab(vocab: Vocab, matrix: np.ndarray,
                  unigram_table_size: int = 10_000_000) -> None:
    """Use the transformation matrix to convert the word vectors.

    Args:
        vocab (Vocab): The Vocab.
        matrix (np.ndarray): The transformation matrix.
        unigram_table_size (int): The unigram table size. Defualts to 10 000 000.
    """
    for d in vocab.vocab.values():
        cvec = d['vec']
        if cvec is None:
            continue
        d['vec'] = convert_vec(cvec, matrix)
    logger.info("Recalc unigram table")
    vocab.make_unigram_table(unigram_table_size)


def convert_context_vectors(cdb: CDB, matrix: np.ndarray) -> None:
    """Use the transformation matrix to convert the context vectors within the CDB.

    Args:
        cdb (CDB): The Context Database.
        matrix (np.ndarray): The transformation matrix.
    """
    for per_cui_dict in cdb.cui2context_vectors.values():
        for type_name, cur_vec in list(per_cui_dict.items()):
            per_cui_dict[type_name] = convert_vec(cur_vec, matrix)
    cdb.is_dirty = True


def convert_vocab_vector_size(cdb: CDB, vocab: Vocab, vec_size: int):
    """Convert the vocab vector size to a smaller one.

    This uses Principal Component Analysis (PCA). The idea is that we
    first center all the word vectors (in Vocab), then compute the
    covariance matrix, then find the eigenvalues and eigenvectors,
    and then we select the top `vec_size` eigenvectors.
    This produces a transformation matrix of shape (vec_size, N),
    where N is the current vector length in the vocab.

    After that, we perform the tranformation. First we transform all
    the vectors in the Vocab. And then we transform all the context
    vectors defined within the CDB.

    NOTE: This requires the CDB as well since the per concept context
    vectors stored within it are based on the vectors in the vocab and
    thus they also need to be transformed.

    Args:
        cdb (CDB): The Concept Database.
        vocab (Vocab): The Vocab.
        vec_size (int): The target vector size.
    """
    logger.info("Converting Vocab and CDB to size %s. Calculating "
                "transformation matrix", vec_size)
    matrix = calc_matrix(vocab, vec_size)
    logger.info("Found transformation matrix with shape %s. "
                "Now converting vocab.", matrix.shape)
    convert_vocab(vocab, matrix)
    logger.info("Done converting vocab, now converting the per concept "
                "context vectors defined in the CDB.")
    convert_context_vectors(cdb, matrix)
    logger.info("Done with the conversion to vocab vector size %s.",
                vec_size)
