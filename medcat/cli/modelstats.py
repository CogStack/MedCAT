from dataclasses import dataclass, field
from typing import Any, List, Dict


@dataclass
class CDBStats:
    number_of_concepts: int = 0
    number_of_names: int = 0
    number_of_concepts_received_training: int = 0
    number_of_seen_training_examples: int = 0
    average_training_example_per_concept: float = 0.0
    ontology_type: List[str] = field(default_factory=list) # type: ignore
    ontology_version: float = 1.0


@dataclass
class VocabStats:
    number_of_words: int = 0


@dataclass
class TrainerStats:
    authors: List[str] = field(default_factory=list) # type: ignore
    epoch: int = 0
    concept_precision: float = 0.0
    concept_f1: float = 0.0
    concept_recall: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    cui_counts: int = 0
    ontology_type: List[str] = field(default_factory=list) # type: ignore
    ontology_version: float = 1.0
    meta_project_data: Dict = field(default_factory=list) # type: ignore


@dataclass
class MetaCATStats:
    precision: float = 0.0
    f1: float = 0.0
    recall: float = 0.0
    learning_rate: float = 0.0
    nepochs: int = 0
    class_report: Dict[Any, Any] = field(default_factory=list) # type: ignore
