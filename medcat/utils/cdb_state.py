import logging
import contextlib
from typing import Dict, TypedDict, Set, List, Protocol, cast
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


def copy_cdb_state(cdb) -> CDBState:
    return cast(CDBState, {
        k: deepcopy(getattr(cdb, k)) for k in CDBState.__annotations__
    })


def save_cdb_state(cdb, file_path: str) -> None:
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
    for k, v in state.items():
        setattr(cdb, k, v)


def load_and_apply_cdb_state(cdb, file_path: str) -> None:
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
    if save_state_to_disk:
        with on_disk_memory_capture(cdb):
            yield
    else:
        with in_memory_state_capture(cdb):
            yield


@contextlib.contextmanager
def in_memory_state_capture(cdb):
    state = copy_cdb_state(cdb)
    yield
    apply_cdb_state(cdb, state)


@contextlib.contextmanager
def on_disk_memory_capture(cdb):
    with tempfile.NamedTemporaryFile() as tf:
        save_cdb_state(cdb, tf.name)
        yield
        load_and_apply_cdb_state(cdb, tf.name)
