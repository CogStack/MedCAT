import os
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, TypeVar, Type
from medcat.cdb import CDB
from medcat.utils.decorators import check_positive

T = TypeVar("T", bound="Checkpoint")


logger = logging.getLogger(__name__) # separate logger from the package-level one


class Checkpoint(object):
    """The base class of checkpoint objects

    Args:
        dir_path (str):
            The path to the parent directory of checkpoint files.
        steps (int):
            The number of processed sentences/documents before a checkpoint is saved
            (N.B.: A small number could result in error "no space left on device"),
        max_to_keep (int):
            The maximum number of checkpoints to keep
            (N.B.: A large number could result in error "no space left on device").
    """
    DEFAULT_STEP = 1000
    DEFAULT_MAX_TO_KEEP = 1

    @check_positive
    def __init__(self, dir_path: str, *, steps: int = DEFAULT_STEP, max_to_keep: int = DEFAULT_MAX_TO_KEEP) -> None:
        self._dir_path = os.path.abspath(dir_path)
        self._steps = steps
        self._max_to_keep = max_to_keep
        self._file_paths: List[str] = []
        self._count = 0
        os.makedirs(self._dir_path, exist_ok=True)

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        check_positive(lambda _: ...)(value)    # [https://github.com/python/mypy/issues/1362]
        self._steps = value

    @property
    def max_to_keep(self) -> int:
        return self._max_to_keep

    @max_to_keep.setter
    def max_to_keep(self, value: int) -> None:
        check_positive(lambda _: ...)(value)    # [https://github.com/python/mypy/issues/1362]
        self._max_to_keep = value

    @property
    def count(self) -> int:
        return self._count

    @property
    def dir_path(self) -> str:
        return self._dir_path

    @classmethod
    def from_latest(cls: Type[T], dir_path: str) -> T:
        """Retrieve the latest checkpoint from the parent directory.

        Args:
            dir_path (string):
                The path to the directory containing checkpoint files.
        Returns:
            T: A new checkpoint object.
        """
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
        logger.info(f"Checkpoint loaded from {latest_ckpt}")
        return checkpoint

    def save(self, cdb: CDB, count: int) -> None:
        """Save the CDB as the latest checkpoint.

        Args:
            cdb (medcat.CDB):
                The MedCAT CDB object to be checkpointed.
            count (count):
                The number of the finished steps.
        """
        ckpt_file_path = os.path.join(os.path.abspath(self._dir_path), "checkpoint-%s-%s" % (self.steps, count))
        while len(self._file_paths) >= self._max_to_keep:
            to_remove = self._file_paths.pop(0)
            os.remove(to_remove)
        cdb.save(ckpt_file_path)
        logger.debug("Checkpoint saved: %s", ckpt_file_path)
        self._file_paths.append(ckpt_file_path)
        self._count = count

    def restore_latest_cdb(self) -> CDB:
        """Restore the CDB from the latest checkpoint.

        Returns:
            cdb (medcat.CDB):
                The MedCAT CDB object.
        """
        if not os.path.isdir(self._dir_path):
            raise Exception("Checkpoints not found. You need to train from scratch.")
        ckpt_file_paths = self._get_ckpt_file_paths(self._dir_path)
        if not ckpt_file_paths:
            raise Exception("Checkpoints not found. You need to train from scratch.")
        latest_ckpt = ckpt_file_paths[-1]
        _, count = self._get_steps_and_count(latest_ckpt)
        self._file_paths = ckpt_file_paths
        self._count = count
        return CDB.load(self._file_paths[-1])

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


@dataclass
class CheckpointConfig(object):
    output_dir: str = "checkpoints"
    steps: int = Checkpoint.DEFAULT_STEP
    max_to_keep: int = Checkpoint.DEFAULT_MAX_TO_KEEP


class CheckpointManager(object):
    """The class for managing checkpoints of specific training type and their configuration

    Args:
        name (str):
            The name of the checkpoint manager (also used as the checkpoint base directory name).
        checkpoint_config (medcat.utils.checkpoint.CheckpointConfig):
            The checkpoint config object.
    """
    def __init__(self, name: str, checkpoint_config: CheckpointConfig) -> None:
        self.name = name
        self.checkpoint_config = checkpoint_config

    def create_checkpoint(self, dir_path: Optional[str] = None) -> "Checkpoint":
        """Create a new checkpoint inside the checkpoint base directory.

        Args:
            dir_path (str):
                The path to the checkpoint directory.

        Returns:
            CheckPoint: A checkpoint object.
        """
        dir_path = dir_path or os.path.join(os.path.abspath(os.getcwd()), self.checkpoint_config.output_dir, self.name, str(int(time.time())))
        return Checkpoint(dir_path,
                          steps=self.checkpoint_config.steps,
                          max_to_keep=self.checkpoint_config.max_to_keep)

    def get_latest_checkpoint(self, base_dir_path: Optional[str] = None) -> "Checkpoint":
        """Retrieve the latest checkpoint from the checkpoint base directory.

        Args:
            base_dir_path (string):
                The path to the directory containing checkpoint files.

        Returns:
            CheckPoint: A checkpoint object
        """
        base_dir_path = base_dir_path or os.path.join(os.path.abspath(os.getcwd()), self.checkpoint_config.output_dir, self.name)
        ckpt_dir_path = self.get_latest_training_dir(base_dir_path=base_dir_path)
        checkpoint = Checkpoint.from_latest(dir_path=ckpt_dir_path)
        checkpoint.steps = self.checkpoint_config.steps
        checkpoint.max_to_keep = self.checkpoint_config.max_to_keep
        return checkpoint

    @classmethod
    def get_latest_training_dir(cls, base_dir_path: str) -> str:
        """Retrieve the latest training directory containing all checkpoints.

        Args:
            base_dir_path (string):
                The path to the directory containing all checkpointed trainings.
        Returns:
            str: The path to the latest training directory containing all checkpoints.
        """
        if not os.path.isdir(base_dir_path):
            raise ValueError(f"Checkpoint folder passed in does not exist: {base_dir_path}")
        ckpt_dir_paths = os.listdir(base_dir_path)
        if not ckpt_dir_paths:
            raise ValueError("No existing training found")
        ckpt_dir_paths.sort()
        ckpt_dir_path = os.path.abspath(os.path.join(base_dir_path, ckpt_dir_paths[-1]))
        return ckpt_dir_path
