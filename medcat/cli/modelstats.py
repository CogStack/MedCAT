from dataclasses import dataclass

@dataclass
class CDBStats:
    number_of_concepts : int = 0
    number_of_names : int = 0
    number_of_concepts_received_training : int = 0
    number_of_seen_training_examples : int = 0
    average_training_example_per_concept : float = 0.0
    ontology_type: str = "SNOMED"
    ontology_version: str = "1.0"

@dataclass
class TrainerStats:
    epoch : int = 0
    concept_precision : float = 0.0
    concept_f1 : float = 0.0
    concept_recall : float = 0.0
    false_positives : int = 0
    false_negatives : int = 0
    true_positives : int = 0
    cui_counts : int = 0
    ontology_type: str = "SNOMED"
    ontology_version: str = "1.0"


@dataclass
class MetaCATStats:
    precision : float = 0.0
    f1 : float = 0.0
    recall : float = 0.0
    learning_rate : float = 0.0
    nepochs : int = 0
    model_name : str = "N/A"
    score_average : str = "N/A"