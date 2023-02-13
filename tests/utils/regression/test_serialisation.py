
import unittest
from medcat.utils.regression.results import ResultDescriptor

from medcat.utils.regression.targeting import CUIWithChildFilter, FilterOptions, FilterStrategy, FilterType
from medcat.utils.regression.targeting import TypedFilter
from medcat.utils.regression.checking import RegressionChecker, RegressionCase, MetaData


class TestSerialisation(unittest.TestCase):

    def test_TypedFilter_serialises(self, ft=FilterType.NAME, vals=['FNAME1', 'FNAME2']):
        tf = TypedFilter(type=ft, values=vals)
        self.assertIsInstance(tf.to_dict(), dict)

    def test_TypedFilter_deserialises(self, ft=FilterType.NAME, vals=['FNAME-1', 'FNAME=2']):
        tf = TypedFilter(type=ft, values=vals)
        tf2 = TypedFilter.from_dict(tf.to_dict())[0]
        self.assertIsInstance(tf2, TypedFilter)

    def test_TypedFilter_deserialises_to_one(self, ft=FilterType.NAME, vals=['FNAME-1', 'FNAME=2']):
        tf = TypedFilter(type=ft, values=vals)
        l = TypedFilter.from_dict(tf.to_dict())
        self.assertEqual(len(l), 1)

    def test_TypedFilter_deserialises_to_same(self, ft=FilterType.NAME, vals=['FNAME-1', 'FNAME=2']):
        tf = TypedFilter(type=ft, values=vals)
        tf2 = TypedFilter.from_dict(tf.to_dict())[0]
        self.assertEqual(tf, tf2)

    def test_CUIWithChildFilter_deserialises_to_same(self, cui='the-cui', depth=5):
        delegate = TypedFilter(type=FilterType.CUI_AND_CHILDREN, values=[cui])
        tf = CUIWithChildFilter(
            type=FilterType.CUI_AND_CHILDREN, depth=depth, delegate=delegate)
        tf2 = TypedFilter.from_dict(tf.to_dict())[0]
        self.assertIsInstance(tf2, CUIWithChildFilter)
        self.assertEqual(tf, tf2)

    def test_multiple_TypedFilter_serialise(self, ft1=FilterType.NAME, ft2=FilterType.CUI, vals1=['NAMEFILTER1'], vals2=['CUI1']):
        tf1 = TypedFilter(type=ft1, values=vals1)
        tf2 = TypedFilter(type=ft2, values=vals2)
        dicts = TypedFilter.list_to_dicts([tf1, tf2])
        self.assertIsInstance(dicts, list)
        self.assertEqual(len(dicts), 2)
        for d in dicts:
            with self.subTest(f'Assuming dict: {d}'):
                self.assertIsInstance(d, dict)

    def test_multiple_TypedFilter_serialise_into(self, ft1=FilterType.NAME, ft2=FilterType.CUI, vals1=['NAMEFILTER1'], vals2=['CUI1']):
        tf1 = TypedFilter(type=ft1, values=vals1)
        tf2 = TypedFilter(type=ft2, values=vals2)
        dicts = TypedFilter.list_to_dicts([tf1, tf2])
        self.assertIsInstance(dicts, list)

    def test_multiple_TypedFilter_deserialise(self, ft1=FilterType.NAME, ft2=FilterType.CUI, vals1=['NAMEFILTER1'], vals2=['CUI1']):
        tf1 = TypedFilter(type=ft1, values=vals1)
        tf2 = TypedFilter(type=ft2, values=vals2)
        tf1_cp, tf2_cp = TypedFilter.from_dict(
            TypedFilter.list_to_dict([tf1, tf2]))
        self.assertIsInstance(tf1_cp, TypedFilter)
        self.assertIsInstance(tf2_cp, TypedFilter)

    def test_multiple_TypedFilter_deserialise_to_same(self, ft1=FilterType.NAME, ft2=FilterType.CUI, vals1=['NAMEFILTER1'], vals2=['CUI1']):
        tf1 = TypedFilter(type=ft1, values=vals1)
        tf2 = TypedFilter(type=ft2, values=vals2)
        the_dict = TypedFilter.list_to_dict([tf1, tf2])
        self.assertIsInstance(the_dict, dict)
        tf1_cp, tf2_cp = TypedFilter.from_dict(the_dict)
        self.assertEqual(tf1, tf1_cp)
        self.assertEqual(tf2, tf2_cp)

    def test_RegressionCase_serialises(self, name='the-name', options=FilterOptions(strategy=FilterStrategy.ALL),
                                       filters=[TypedFilter(
                                           type=FilterType.NAME, values=['nom1', 'nom2'])],
                                       phrases=['the %s phrase']):
        rc = RegressionCase(name=name, options=options,
                            filters=filters, phrases=phrases, report=ResultDescriptor(name=name))
        self.assertIsInstance(rc.to_dict(), dict)

    def test_RegressionCase_deserialises_to_same(self, name='the-name', options=FilterOptions(strategy=FilterStrategy.ANY),
                                                 filters=[TypedFilter(
                                                     type=FilterType.NAME, values=['nom1', 'nom2'])],
                                                 phrases=['the %s phrase']):
        rc = RegressionCase(name=name, options=options,
                            filters=filters, phrases=phrases, report=ResultDescriptor(name=name))
        rc2 = RegressionCase.from_dict(name, rc.to_dict())
        self.assertIsInstance(rc2, RegressionCase)
        self.assertEqual(rc, rc2)

    def test_RegressionChecker_serialises(self, name='the-name', options=FilterOptions(strategy=FilterStrategy.ALL),
                                          filters=[TypedFilter(
                                              type=FilterType.NAME, values=['nom1', 'nom2'])],
                                          phrases=['the %s phrase']):
        rc = RegressionCase(name=name, options=options,
                            filters=filters, phrases=phrases, report=ResultDescriptor(name=name))
        checker = RegressionChecker(cases=[rc], metadata=MetaData.unknown())
        self.assertIsInstance(checker.to_dict(), dict)

    def test_RegressionChecker_deserialises_to_same(self, name='the-name', options=FilterOptions(strategy=FilterStrategy.ANY),
                                                    filters=[TypedFilter(
                                                        type=FilterType.NAME, values=['nom1', 'nom2'])],
                                                    phrases=['the %s phrase']):
        rc = RegressionCase(name=name, options=options,
                            filters=filters, phrases=phrases, report=ResultDescriptor(name=name))
        checker = RegressionChecker(cases=[rc], metadata=MetaData.unknown())
        checker2 = RegressionChecker.from_dict(checker.to_dict())
        self.assertIsInstance(checker2, RegressionChecker)
        rc.__eq__
        self.assertEqual(checker, checker2)
