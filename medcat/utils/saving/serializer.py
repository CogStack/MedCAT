"""This modlue is responsible for the (new) methods of saving and loading parts of MedCAT.

The idea is to move away from saving medcat files using the dill/pickle.
And to save as well as load them in some other way.
"""
import os
from typing import cast, Dict, Optional  # , List, Set, Tuple
import dill
import shelve
# import marshal
import json
# from scipy.io import loadmat, savemat

# import pydantic

from medcat.cdb import CDB
from medcat.config import Config


_SAVEMAT_MAX_CHAR = 31


__SPECIALITY_NAMES_CUI = set(["cui2names", "cui2snames", "cui2type_ids"])
__SPECIALITY_NAMES_NAME = set(
    ["name2cuis", "name2cuis2status", "name_isupper"])
__SPECIALITY_NAMES_OTHER = set()  # none for now
__SPECIALITY_NAMES_BUILDABLE = set(["snames"])  # these are ignored altogether
SPECIALITY_NAMES = __SPECIALITY_NAMES_CUI | __SPECIALITY_NAMES_NAME | __SPECIALITY_NAMES_OTHER | __SPECIALITY_NAMES_BUILDABLE
__SPECIALITY_SAVE_NAMES = dict(
    (name, name[:_SAVEMAT_MAX_CHAR]) for name in SPECIALITY_NAMES)
# __SPECIALITY_LOAD_NAMES = dict((val, key)
#                                for key, val in __SPECIALITY_SAVE_NAMES.items())
if len(set(__SPECIALITY_SAVE_NAMES.values())) != len(__SPECIALITY_SAVE_NAMES):
    raise ValueError("Some ambiguous names would be saved")


class SetEncode(json.JSONEncoder):
    SET_IDENTIFIER = '==SET=='

    def default(self, obj):
        if isinstance(obj, set):
            return {SetEncode.SET_IDENTIFIER: list(obj)}
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def set_decode(dct: dict) -> dict:
        if SetEncode.SET_IDENTIFIER in dct:
            return set(dct[SetEncode.SET_IDENTIFIER])
        return dct


class JsonSetSerializer:
    """Dumper for special case"""

    def __init__(self, folder: str, name: str) -> None:
        self.name = name
        self.file_name = os.path.join(folder, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        elif not os.path.isdir(folder):
            raise ValueError(f'Folder expected, got fille: {folder}')
        if os.path.isdir(self.file_name):
            raise ValueError(
                f'Expected file, found folder: {self.file_name}')
        self.shelves: Dict[str, shelve.Shelf] = {}

    def write(self, d: dict) -> None:
        with open(self.file_name, 'w') as f:
            json.dump(d, f, cls=SetEncode)

    def read(self) -> dict:
        with open(self.file_name, 'r') as f:
            # data = json.load(f, cls=SetDecode)
            data = json.load(f, object_hook=SetEncode.set_decode)
        return data


class CDBSerializer:

    def __init__(self, main_path: str, json_path: str = None) -> None:
        """_summary_

        Args:
            path (str): The raw file path
            shelve_path (str, optional): The shelf path for more memory intensive parts. 
                Defaults to None, in which case everything is expected to be within the path.
        """
        # self.has_shelf = shelve_path is not None
        self.main_path = main_path
        self.json_path = json_path
        self.jsons: Optional[Dict[str, JsonSetSerializer]] = {}
        if self.json_path is not None:
            for name in SPECIALITY_NAMES:
                self.jsons[name] = JsonSetSerializer(self.json_path, name)
        else:
            self.jsons = None

    def serialize(self, cdb: CDB, overwrite: bool = False) -> None:
        """Used to dump CDB to a file or two files.

        This will generally use dill.dump for most things.
        However, some parts are taken care of separately
        since they take up the vast majority of the space on disk
        and time on read.
        These are:
            - name2cuis
            - name2cuis2status
            - snames
            - cui2names
            - cui2snames
            - cui2type_ids
            - name_isupper
        # TODO - describe how they are dealt with, in some way

        Args:
            cdb (CDB): The context database (CDB)
            target (str): The file to serialize to
            overwrite (bool, optional): _description_. Defaults to False.
        """
        if not overwrite and os.path.exists(self.main_path):
            raise ValueError(
                f'Cannot overwrite file "{self.main_path}" - specify overwrite=True if you wish to overwrite')
        if self.jsons is not None and os.path.exists(self.json_path) and not overwrite:
            raise ValueError(f'Unable to overwrite shelf path "{self.json_path}"'
                             ' - specify overrwrite=True if you wish to overwrite')
        to_save = {}
        to_save['config'] = cdb.config.asdict()
        to_save['cdb_main' if self.jsons is not None else 'cdb'] = dict(
            ((key, val) for key, val in cdb.__dict__.items() if
             key != 'config' and
             (self.jsons is None or key not in SPECIALITY_NAMES)))
        with open(self.main_path, 'wb') as f:
            dill.dump(to_save, f)
        if self.jsons is not None:
            for name in SPECIALITY_NAMES:
                self.jsons[name].write(cdb.__dict__[name])

    def deserialize(self) -> CDB:
        with open(self.main_path, 'rb') as f:
            data = dill.load(f)
        config = cast(Config, Config.from_dict(data['config']))
        cdb = CDB(config=config)
        if self.jsons is None:
            cdb_main = data['cdb']
        else:
            cdb_main = data['cdb_main']
        cdb.__dict__.update(cdb_main)
        if self.jsons is not None:
            for name in SPECIALITY_NAMES:
                cdb.__dict__[name] = self.jsons[name].read()
        return cdb
