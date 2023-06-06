"""This modlue is responsible for the (new) methods of saving and loading parts of MedCAT.

The idea is to move away from saving medcat files using the dill/pickle.
And to save as well as load them in some other way.
"""
import os
import logging
from typing import cast, Dict, Optional, Type
import dill
import json

from medcat.config import Config
from medcat.utils.saving.coding import CustomDelegatingEncoder, default_hook, default_postprocessing

logger = logging.getLogger(__name__)


__SPECIALITY_NAMES_CUI = set(["cui2names", "cui2snames", "cui2type_ids"])
__SPECIALITY_NAMES_NAME = set(
    ["name2cuis", "name2cuis2status", "name_isupper"])
__SPECIALITY_NAMES_OTHER = set(["snames", "addl_info"])
ONE2MANY = set(['cui2many', 'name2many'])  # these may or may not exist
SPECIALITY_NAMES = __SPECIALITY_NAMES_CUI | __SPECIALITY_NAMES_NAME | __SPECIALITY_NAMES_OTHER | ONE2MANY


class JsonSetSerializer:
    """JSON serializer with set comprehension.

    This serializer allows serializing and deserializing sets through JSON"""

    def __init__(self, folder: str, name: str) -> None:
        self.name = name
        self.file_name = os.path.join(folder, name)
        if not self.file_name.endswith('.json'):
            self.file_name = self.file_name + '.json'
        if not os.path.exists(folder):
            os.makedirs(folder)
        elif not os.path.isdir(folder):
            raise ValueError(f'Folder expected, got fille: {folder}')
        if os.path.isdir(self.file_name):
            raise ValueError(
                f'Expected file, found folder: {self.file_name}')

    def write(self, d: dict) -> None:
        """Write the specified dictionary to the this serializer's file.

        Args:
            d (dict): The dict to write on file.
        """
        logger.info('Writing data for "%s" into "%s"',
                    self.name, self.file_name)
        with open(self.file_name, 'w') as f:
            # the def_inst method, when called,
            # returns the right type of object anyway

            json.dump(d, f, cls=cast(Type[json.JSONEncoder],
                                     CustomDelegatingEncoder.def_inst))

    def read(self) -> dict:
        """Read the json file specified by this serializer.

        Returns:
            dict: The dict represented by this json file.
        """
        logger.info('Reading data for %s from %s', self.name, self.file_name)
        with open(self.file_name, 'r') as f:
            data = json.load(
                f, object_hook=default_hook)
        return data


class CDBSerializer:
    """A (potentially) semi-JSON based serializer for CDB.

    The parts that take up the most space within a CDB can be saved in JSON files.
    That is the following attributes of a CDB:
        - name2cuis
        - name2cuis2status
        - snames
        - cui2names
        - cui2snames
        - cui2type_ids
        - name_isupper
        - addl_info
    These are specified at the top of the module (in `SPECIALITY_NAMES`).

    The rest of the information (i.e config and other less memory intensive parts) will
    still be saved using dill like they have been before.

    The objects of this class can be used for both serializing as well as deserializing.
    If the `json_path` parameter is passed, the JSON (de)serialization will be performed.

    Args:
        main_path (str): The path for the main part (i.e config and other less memory intensive parts)
        json_path (str, optional): The JSON. Defaults to None.
    """

    def __init__(self, main_path: str, json_path: Optional[str] = None) -> None:
        self.main_path = main_path
        self.json_path = json_path
        self.jsons: Optional[Dict[str, JsonSetSerializer]] = {}
        if self.json_path is not None:
            for name in SPECIALITY_NAMES:
                self.jsons[name] = JsonSetSerializer(self.json_path, name)
        else:
            self.jsons = None

    def serialize(self, cdb, overwrite: bool = False) -> None:
        """Used to dump CDB to a file or or multiple files.

        If `json_path` was specified to the constructor, this will serialize
        some of the parts that take up more memory in JSON files in said directory.
        In that case, the rest of the info is saved into the `main_path` passed to
        the consturctor
        Otherwise, everything is saved to the `main_path` using `dill.dump`
        just like in previous cases.

        Args:
            cdb (CDB): The context database (CDB)
            overwrite (bool, optional): Whether to allow overwriting existing files. Defaults to False.

        Raises:
            ValueError: If file(s) exist(s) and overwrite if `False`
        """
        if not overwrite and os.path.exists(self.main_path):
            raise ValueError(
                f'Cannot overwrite file "{self.main_path}" - specify overwrite=True if you wish to overwrite')
        if self.jsons is not None:
            for name in SPECIALITY_NAMES:
                ser = self.jsons[name]
                if not overwrite and os.path.exists(ser.file_name):
                    raise ValueError(
                        f'Cannot overwrite file {ser.file_name} - specify overwrite=True if you wish to overwrite')
        if self.json_path and os.path.exists(self.json_path) and not overwrite:
            raise ValueError(f'Unable to overwrite shelf path "{self.json_path}"'
                             ' - specify overrwrite=True if you wish to overwrite')
        to_save = {}
        to_save['config'] = cdb.config.asdict()
        # This uses different names so as to not be ambiguous
        # when looking at files whether the json parts should
        # exist separately or not
        to_save['cdb_main' if self.jsons is not None else 'cdb'] = dict(
            ((key, val) for key, val in cdb.__dict__.items() if
             key != 'config' and
             (self.jsons is None or key not in SPECIALITY_NAMES)))
        logger.info('Dumping CDB to %s', self.main_path)
        with open(self.main_path, 'wb') as f:
            dill.dump(to_save, f)
        if self.jsons is not None:
            for name in SPECIALITY_NAMES:
                if name not in cdb.__dict__:
                    continue  # in case cui2many doesn't exit
                self.jsons[name].write(cdb.__dict__[name])

    def deserialize(self, cdb_cls):
        """Deserializes the json in the specified file info a CDB.

        If the `json_path` was specified to the constructor,
        the JSON serialized files are used.
        Otherwise, everything is loaded from the `main_path` file.

        Returns:
            CDB: The resulting CDB.
        """
        logger.info('Reading CDB data from %s', self.main_path)
        with open(self.main_path, 'rb') as f:
            data = dill.load(f)
        config = cast(Config, Config.from_dict(data['config']))
        cdb = cdb_cls(config=config)
        if self.jsons is None:
            cdb_main = data['cdb']
        else:
            cdb_main = data['cdb_main']

        # Load data into the new cdb instance
        for k in cdb.__dict__:
            if k in cdb_main:
                cdb.__dict__[k] = cdb_main[k]

        # Load data into new CDB from additional JSON files
        # if applicable
        if self.jsons is not None:
            for name in SPECIALITY_NAMES:
                if not os.path.exists(self.jsons[name].file_name):
                    continue  # in case of non-memory-optimised where cui2many doesn't exist
                cdb.__dict__[name] = self.jsons[name].read()
        # if anything has
        # been registered to postprocess the CDBs
        default_postprocessing(cdb)
        return cdb
