import logging
import contextlib
from typing import Dict, TypedDict, Set, List, cast
import numpy as np
import tempfile
import dill

from copy import deepcopy



logger = logging.getLogger(__name__) # separate logger from the package-level one


CDBState = TypedDict(
    'CDBState',
    {
        'name2cuis': Dict[str, List[str]],
        'snames': Set[str],
        'cui2names': Dict[str, Set[str]],
        'cui2snames': Dict[str, Set[str]],
        'cui2context_vectors': Dict[str, Dict[str, np.ndarray]],
        'cui2count_train': Dict[str, int],
        'name_isupper': Dict,
        'vocab': Dict[str, int],
    })
"""CDB State.

This is a dictionary of the parts of the CDB that change during
(supervised) training. It can be used to store and restore the
state of a CDB after modifying it.

Currently, the following fields are saved:
 - name2cuis
 - snames
 - cui2names
 - cui2snames
 - cui2context_vectors
 - cui2count_train
 - name_isupper
 - vocab
"""


def copy_cdb_state(cdb) -> CDBState:
    """Creates a (deep) copy of the CDB state.

    Grabs the fields that correspond to the state,
    creates deep copies, and returns the copies.

    Args:
        cdb: The CDB from which to grab the state.

    Returns:
        CDBState: The copied state.
    """
    return cast(CDBState, {
        k: deepcopy(getattr(cdb, k)) for k in CDBState.__annotations__
    })


def save_cdb_state(cdb, file_path: str) -> None:
    """Saves CDB state in a file.

    Currently uses `dill.dump` to save the relevant fields/values.

    Args:
        cdb: The CDB from which to grab the state.
        file_path (str): The file to dump the state.
    """
    # NOTE: The difference is that we don't create a copy here.
    #       That is so that we don't have to occupy the memory for
    #       both copies
    the_dict = {
        k: getattr(cdb, k) for k in CDBState.__annotations__
    }
    logger.debug("Saving CDB state on disk at: '%s'", file_path)
    with open(file_path, 'wb') as f:
        dill.dump(the_dict, f)


def apply_cdb_state(cdb, state: CDBState) -> None:
    """Apply the specified state to the specified CDB.

    This overwrites the current state of the CDB with one provided.

    Args:
        cdb: The CDB to apply the state to.
        state (CDBState): The state to use.
    """
    for k, v in state.items():
        setattr(cdb, k, v)


def load_and_apply_cdb_state(cdb, file_path: str) -> None:
    """Delete current CDB state and apply CDB state from file.

    This first deletes the current state of the CDB.
    This is to save memory. The idea is that saving the staet
    on disk will save on RAM usage. But it wouldn't really
    work too well if upon load, two instances were still in
    memory.

    Args:
        cdb: The CDB to apply the state to.
        file_path (str): The file where the state has been saved to.
    """
    # clear existing data on CDB
    # this is so that we don't occupy the memory for both the loaded
    # and the on-CDB data
    logger.debug("Clearing CDB state in memory")
    for k in CDBState.__annotations__:
        val = getattr(cdb, k)
        setattr(cdb, k, None)
        del val
    logger.debug("Loading CDB state from disk from '%s'", file_path)
    with open(file_path, 'rb') as f:
        data = dill.load(f)
    for k in CDBState.__annotations__:
        setattr(cdb, k, data[k])


@contextlib.contextmanager
def captured_state_cdb(cdb, save_state_to_disk: bool = False):
    """A context manager that captures and re-applies the initial CDB state.

    The context manager captures/copies the initial state of the CDB when entering.
    It then allows the user to modify the state (i.e training).
    Upon exit re-applies the initial CDB state.

    If RAM is an issue, it is recommended to use `save_state_to_disk`.
    Otherwise the copy of the original state will be held in memory.
    If saved on disk, a temporary file is used and removed afterwards.

    Args:
        cdb: The CDB to use.
        save_state_to_disk (bool): Whether to save state on disk or hold in in memory.
            Defaults to False.

    Yields:
        None
    """
    if save_state_to_disk:
        with on_disk_memory_capture(cdb):
            yield
    else:
        with in_memory_state_capture(cdb):
            yield


@contextlib.contextmanager
def in_memory_state_capture(cdb):
    """Capture the CDB state in memory.

    Args:
        cdb: The CDB to use.

    Yields:
        None
    """
    state = copy_cdb_state(cdb)
    yield
    apply_cdb_state(cdb, state)


@contextlib.contextmanager
def on_disk_memory_capture(cdb):
    """Capture the CDB state in a temporary file.

    Args:
        cdb: The CDB to use

    Yields:
        None
    """
    with tempfile.NamedTemporaryFile() as tf:
        save_cdb_state(cdb, tf.name)
        yield
        load_and_apply_cdb_state(cdb, tf.name)
