import unittest
from medcat.idconfig import BaseConfig

# from pydantic import BaseModel

# class BCPBM(BaseModel):
#     attr: str = 'DEFAULT'


# def check_diffs(d1: dict, d2: dict, recursive=True):
#     for k1, k2 in zip(d1.keys(), d2.keys()):
#         if k1 != k2:
#             print('DIFFERENT', k1, 'vs', k2)
#             print(d1[k1])
#             print('vs')
#             print(d2[k2])
#             continue # TODO - do I want to continue regardless?
#         v1, v2 = d1[k1], d2[k2]
#         if v1 != v2:
#             print('Different values for', k1, k2)
#             print(type(v1), 'vs', type(v2), 'or', id(v1), 'vs', id(v2), v1==v2, v1!=v2)
#             print(v1)
#             print('vs')
#             print(v2)
#             if recursive and type(v1) == type(v2) == dict:
#                 print('GOING DEEPER')
#                 check_diffs(v1, v2)

class BaseConfigTests(unittest.TestCase):

    def test_creates_new_defaults(self):
        # the default values are mutable
        # so if new instances weren't being created,
        # the second one would refer to the same instance
        # for the .general field as the first and as such
        # this test would fail
        bc1 = BaseConfig()
        bc1.general.spacy_model = 'smth'
        bc2 = BaseConfig()
        print(bc2.general.spacy_model, 'vs', bc1.general.spacy_model)
        self.assertNotEqual(bc2.general.spacy_model, bc1.general.spacy_model)

    def test_uses_different_internal_instances(self):
        # again, if the instance for .general was the same,
        # we would see this test fail
        bc1: BaseConfig = BaseConfig()
        bc2: BaseConfig = BaseConfig()
        self.assertEqual(bc1, bc2)
        bc1.general.spacy_model = 'Non-default-value'
        self.assertNotEqual(bc1.general, bc2.general)
        self.assertNotEqual(bc1.general.spacy_model, bc2.general.spacy_model)
        self.assertNotEqual(bc1, bc2)

