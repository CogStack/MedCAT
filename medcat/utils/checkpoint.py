import os
import shutil
import logging
import jsonpickle
from typing import List, Tuple, Optional, Dict
from medcat.cdb import CDB
from medcat.utils.decorators import check_positive


class Checkpoint(object):

    jsonpickle.set_encoder_options('json', sort_keys=True, indent=2)
    log = logging.getLogger(__package__)

    @check_positive
    def __init__(self, dir_path: str, *, steps: int = 1000, max_to_keep: int = 1) -> None:
        """ Initialise the checkpoint object
        Args:
            dir_path (str):
                The path to the checkpoint directory.
            steps (int):
                The number of processed sentences/documents before a checkpoint is saved.
                N.B.: A small number could result in error "no space left on device".
            max_to_keep (int):
                The maximum number of checkpoints to keep.
                N.B.: A large number could result in error "no space left on device".
        """
        self._dir_path = os.path.abspath(dir_path)
        self._steps = steps
        self._max_to_keep = max_to_keep
        self._file_paths: List[str] = []
        self._count = 0
        os.makedirs(self._dir_path, exist_ok=True)

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter  # type: ignore
    # [https://github.com/python/mypy/issues/1362]
    @check_positive
    def steps(self, value: int) -> None:
        self._steps = value

    @property
    def max_to_keep(self) -> int:
        return self._max_to_keep

    @max_to_keep.setter  # type: ignore
    # [https://github.com/python/mypy/issues/1362]
    @check_positive
    def max_to_keep(self, value: int) -> None:
        self._max_to_keep = value

    @property
    def count(self) -> int:
        return self._count

    @classmethod
    def restore(cls, dir_path: str) -> "Checkpoint":
        if not os.path.isdir(dir_path):
            raise Exception("Checkpoints not found. You need to train from scratch.")
        ckpt_file_paths = cls._get_ckpt_file_paths(dir_path)
        if not ckpt_file_paths:
            raise Exception("Checkpoints not found. You need to train from scratch.")
        latest_ckpt = ckpt_file_paths[-1]
        steps, count = cls._get_steps_and_count(latest_ckpt)
        checkpoint = cls(dir_path, steps=steps)
        checkpoint._file_paths = ckpt_file_paths
        checkpoint._count = count
        return checkpoint

    def purge(self) -> None:
        shutil.rmtree(self._dir_path)
        os.makedirs(self._dir_path)
        self._file_paths = []
        self._count = 0

    def save(self, cdb: CDB, count: int) -> None:
        ckpt_file_path = os.path.join(os.path.abspath(self._dir_path), "checkpoint-%s-%s" % (self.steps, count))
        while len(self._file_paths) >= self._max_to_keep:
            to_remove = self._file_paths.pop(0)
            os.remove(to_remove)
        cdb.save(ckpt_file_path)
        self.log.info("Checkpoint saved: %s", ckpt_file_path)
        self._file_paths.append(ckpt_file_path)
        self._count = count

    async def save_async(self, cdb: CDB, count: int) -> None:
        ckpt_file_path = os.path.join(os.path.abspath(self._dir_path), "checkpoint-%s-%s" % (self.steps, count))
        await cdb.save_async(ckpt_file_path)
        self.log.info("Checkpoint saved: %s", ckpt_file_path)
        self._file_paths.append(ckpt_file_path)
        self._file_paths.sort(key=lambda f: self._get_steps_and_count(f)[1])
        self._count = count
        while len(self._file_paths) > self._max_to_keep:
            to_remove = self._file_paths.pop(0)
            os.remove(to_remove)

    def populate(self, cdb: CDB) -> None:
        if not self._file_paths:
            raise Exception("Checkpoints not found. You need to restore or train from scratch.")
        cdb.load(self._file_paths[-1])

    @staticmethod
    def _get_ckpt_file_paths(dir_path: str) -> List[str]:
        ckpt_file_paths = [os.path.abspath(os.path.join(dir_path, f)) for f in os.listdir(dir_path)]
        ckpt_file_paths = [f for f in ckpt_file_paths if os.path.isfile(f) and "checkpoint-" in f]
        if ckpt_file_paths:
            ckpt_file_paths.sort(key=lambda f: Checkpoint._get_steps_and_count(f)[1])
        return ckpt_file_paths

    @staticmethod
    def _get_steps_and_count(file_path) -> Tuple[int, int]:
        file_name_parts = os.path.basename(file_path).split('-')
        return int(file_name_parts[1]), int(file_name_parts[2])
