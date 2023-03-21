import unittest
from medcat.utils.model_creator import create_models
from pathlib import Path


class EntityLinkingTest(unittest.TestCase):
    """Test entity linking after creating MedCAT CDB and Vocab models.

    During class setup, MedCAT CDB and Vocabulary models are generated from included test data. Subsequently, these
    models are assessed using various entity linking scenarios.
    """

    @classmethod
    def setUpClass(cls):
        model_creator_config_path = Path('tests/model_creator/config_example.yml')
        cls.cat = create_models(model_creator_config_path)

    def assert_linked_entities(self, doc, expected_entities, unexpected_entities=None):
        """General assertion function to assess linked entities.

        Function to assess whether a doc-object's linked entities are in concordance with a list of expected entities.

        Args:
            doc (spacy.tokens.Doc):
                A spaCy doc-object containing entities and their respective CUIs as provided by MedCAT
            expected_entities (list):
                List of CUIs that should be in the doc object.
            unexpected_entities (list):
                List of CUIs that should not be in the doc object.
        """
        # Extract found entities
        linked_entities = [entity._.cui for entity in doc.ents]

        # Assert whether expected entities are found
        for expected_entity in expected_entities:
            self.assertIn(expected_entity, linked_entities)

        # Assert whether unexpected entities are not found
        if unexpected_entities is not None:
            for unexpected_entity in unexpected_entities:
                self.assertNotIn(unexpected_entity, linked_entities)


class TestEntityLinking(EntityLinkingTest):
    """Test entity linking.

    Test the created medcat models, as well as the entity linking method, by assessing whether a number of expected
    entities are found in a text.
    """

    def test_entity_linking(self):
        text = "Common treatments include surgery, chemotherapy, and radiotherapy. NSCLC is sometimes treated with " \
               "surgery, whereas SCLC usually responds better to chemotherapy and radiotherapy. Of all people with " \
               "lung cancer in the US, around 17% to 20% survive for at least five years after diagnosis."

        # These entities should be found in above text
        expected_entities = ['C1522449',  # surgery
                             'C3665472',  # chemotherapy
                             'C1522449',  # radiotherapy
                             'C0007131',  # NSCLC
                             'C0149925',  # SCLC
                             'C0242379']  # lung cancer
        unexpected_entities = ['C0006826']  # cancer
        doc = self.cat(text)
        self.assert_linked_entities(doc, expected_entities, unexpected_entities)


class TestLemmatization(EntityLinkingTest):
    """Test lemmatization.

    Some entities should be lemmatized before they can be linked to an entity. The minimal number of
    characters for a word required to be lemmatized is defined in the "min_len_normalize" configuration.
    """

    def test_lemmatization_not_required(self):
        expected_entities = ['C0085639']  # Fall
        text = "In the event of a fall"
        doc = self.cat(text)
        self.assert_linked_entities(doc, expected_entities)

    def test_lemmatization_required(self):
        expected_entities = ['C0085639']  # Fall
        text = "The patient is afraid of falling."
        doc = self.cat(text)
        self.assert_linked_entities(doc, expected_entities)


class TestDiacritics(EntityLinkingTest):
    """Test handling of diacritics.

    Diacritics are common in many non-English languages and require specific functionality in MedCAT. This
    functionality can be enabled via the "diacritics" configuration.
    """

    def test_diacritics_in_cdb(self):
        self.assertIn('ménière', self.cat.cdb.cui2snames['C0025281'])

    def test_diacritics_in_text(self):
        expected_entities = ['C0025281']
        text = "Ménière's disease (MD) is a disorder of the inner ear that is characterized by episodes of vertigo, " \
               "tinnitus, hearing loss, and a fullness in the ear."
        doc = self.cat(text)
        self.assert_linked_entities(doc, expected_entities)


class TestCheckUpperCaseNames(EntityLinkingTest):
    """ Test the "check_upper_case_names" functionality.

    Some capitalized words, often acronyms, have a different meaning when written in lowercase.
    Distinguishing these words can be done via by using the check_upper_case_names functionality, which can be
    enabled via the "check_upper_case_names" configuration.
    """

    def test_uppercase_abbreviation_in_sentence(self):
        text = "Phosphorylation of the MAP has an effect."
        expected_entities = ['C0026045']  # MAP (Microtubule-Associated Proteins)
        doc = self.cat(text)
        self.assert_linked_entities(doc, expected_entities)

    def test_uppercase_abbreviation_start_sentence(self):
        text = "MAP groups have been identified."
        expected_entities = ['C0026045']  # MAP (Microtubule-Associated Proteins)
        doc = self.cat(text)
        self.assert_linked_entities(doc, expected_entities)

    def test_lowercase_non_abbreviation_in_sentence(self):
        text = "MedCAT should not map this name to a medical concept."
        doc = self.cat(text)
        self.assertEqual(len(doc.ents), 0)

    def test_lowercase_non_abbreviation_start_sentence(self):
        text = "Map should not be linked to a medical concept."
        doc = self.cat(text)
        self.assertEqual(len(doc.ents), 0)

    def test_uppercase_non_abbreviation(self):
        text = "TAXOTERE is a drug used to treat certain types of cancer."
        expected_entities = ['C0699967']
        doc = self.cat(text)
        self.assert_linked_entities(doc, expected_entities)


if __name__ == '__main__':
    unittest.main()
