import unittest
import tempfile
import json
import os
import shutil
from medcat.cdb import CDB
from medcat.ner.transformers_ner import TransformersNER
from medcat.config_transformers_ner import ConfigTransformersNER

class TestTransformersNER(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Create results dir for training outputs
        self.results_dir = './results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create a minimal CDB
        self.cdb = CDB()
        
        # Create initial training data with 2 labels and multiple examples
        self.initial_data = {
            "projects": [{
                "documents": [
                    {
                        "text": "Patient has diabetes and hypertension.",
                        "annotations": [
                            {
                                "cui": "C0011849",  # Diabetes
                                "start": 14,
                                "end": 22,
                                "value": "diabetes"
                            },
                            {
                                "cui": "C0020538",  # Hypertension
                                "start": 27,
                                "end": 39,
                                "value": "hypertension"
                            }
                        ]
                    },
                    {
                        "text": "History of diabetes with hypertension.",
                        "annotations": [
                            {
                                "cui": "C0011849",  # Diabetes
                                "start": 12,
                                "end": 20,
                                "value": "diabetes"
                            },
                            {
                                "cui": "C0020538",  # Hypertension
                                "start": 26,
                                "end": 38,
                                "value": "hypertension"
                            }
                        ]
                    },
                    {
                        "text": "Diagnosed with hypertension and diabetes.",
                        "annotations": [
                            {
                                "cui": "C0020538",  # Hypertension
                                "start": 15,
                                "end": 27,
                                "value": "hypertension"
                            },
                            {
                                "cui": "C0011849",  # Diabetes
                                "start": 32,
                                "end": 40,
                                "value": "diabetes"
                            }
                        ]
                    }
                ]
            }]
        }
        
        # Create new training data with an extra label
        self.new_data = {
            "projects": [{
                "documents": [
                    {
                        "text": "Patient has diabetes, hypertension, and asthma.",
                        "annotations": [
                            {
                                "cui": "C0011849",  # Diabetes
                                "start": 14,
                                "end": 22,
                                "value": "diabetes"
                            },
                            {
                                "cui": "C0020538",  # Hypertension
                                "start": 24,
                                "end": 36,
                                "value": "hypertension"
                            },
                            {
                                "cui": "C0004096",  # Asthma
                                "start": 42,
                                "end": 48,
                                "value": "asthma"
                            }
                        ]
                    },
                    {
                        "text": "History of asthma with diabetes and hypertension.",
                        "annotations": [
                            {
                                "cui": "C0004096",  # Asthma
                                "start": 12,
                                "end": 18,
                                "value": "asthma"
                            },
                            {
                                "cui": "C0011849",  # Diabetes
                                "start": 24,
                                "end": 32,
                                "value": "diabetes"
                            },
                            {
                                "cui": "C0020538",  # Hypertension
                                "start": 37,
                                "end": 49,
                                "value": "hypertension"
                            }
                        ]
                    },
                    {
                        "text": "Diagnosed with asthma, diabetes, and hypertension.",
                        "annotations": [
                            {
                                "cui": "C0004096",  # Asthma
                                "start": 15,
                                "end": 21,
                                "value": "asthma"
                            },
                            {
                                "cui": "C0011849",  # Diabetes
                                "start": 23,
                                "end": 31,
                                "value": "diabetes"
                            },
                            {
                                "cui": "C0020538",  # Hypertension
                                "start": 37,
                                "end": 49,
                                "value": "hypertension"
                            }
                        ]
                    }
                ]
            }]
        }
        
        # Save initial training data
        self.initial_data_path = os.path.join(self.tmp_dir.name, 'initial_data.json')
        with open(self.initial_data_path, 'w') as f:
            json.dump(self.initial_data, f)
            
        # Save new training data
        self.new_data_path = os.path.join(self.tmp_dir.name, 'new_data.json')
        with open(self.new_data_path, 'w') as f:
            json.dump(self.new_data, f)

    def tearDown(self):
        # Clean up the temporary directory
        self.tmp_dir.cleanup()
        # Clean up results directory if it exists
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        # Clean up logs directory if it exists
        if os.path.exists('./logs'):
            shutil.rmtree('./logs')

    def test_ignore_extra_labels(self):
        # Create and train initial model with tiny BERT
        config = ConfigTransformersNER()
        config.general['model_name'] = 'prajjwal1/bert-tiny'
        # Set to single epoch and small test size for faster testing
        config.general['num_train_epochs'] = 1
        config.general['test_size'] = 0.1
        
        # Create training arguments with reduced epochs
        from transformers import TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.results_dir,  # Use the class results_dir
            num_train_epochs=1
        )
        
        ner = TransformersNER(self.cdb, config=config, training_arguments=training_args)
        ner.train(self.initial_data_path)
        
        # Save the model
        model_path = os.path.join(self.tmp_dir.name, 'model')
        ner.save(model_path)
        
        # Load the saved model
        loaded_ner = TransformersNER.load(model_path)
        
        # Get initial number of labels
        initial_num_labels = len(loaded_ner.tokenizer.label_map)
        
        # Train with ignore_extra_labels=True
        loaded_ner.train(self.new_data_path, ignore_extra_labels=True)
        
        # Verify number of labels hasn't changed
        self.assertEqual(
            len(loaded_ner.tokenizer.label_map),
            initial_num_labels,
            "Number of labels changed despite ignore_extra_labels=True"
        )
        
        # Verify only original labels are present (including special tokens)
        expected_labels = {"C0011849", "C0020538", "O", "X"}
        self.assertEqual(
            set(loaded_ner.tokenizer.label_map.keys()),
            expected_labels,
            "Label map contains unexpected labels"
        )
        
        # Train with ignore_extra_labels=False
        loaded_ner.train(self.new_data_path, ignore_extra_labels=False)
        
        # Verify new label was added
        self.assertEqual(
            len(loaded_ner.tokenizer.label_map),
            initial_num_labels + 1,
            "New label was not added when ignore_extra_labels=False"
        )
        
        # Verify all labels are present (including special tokens)
        expected_labels = {"C0011849", "C0020538", "C0004096", "O", "X"}
        self.assertEqual(
            set(loaded_ner.tokenizer.label_map.keys()),
            expected_labels,
            "Label map missing expected labels"
        )

if __name__ == '__main__':
    unittest.main() 