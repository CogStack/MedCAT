import argparse
import logging
import yaml

from medcat.cdb_maker import CDBMaker
from medcat.utils.make_vocab import MakeVocab
from medcat.cat import CAT
from medcat.config import Config
from pathlib import Path

DEFAULT_UNIGRAM_TABLE_SIZE = 100000000

logger = logging.getLogger(__package__)


def create_cdb(concept_csv_file, medcat_config):
    """Create concept database from csv.

    Args:
        concept_csv_file (pathlib.Path):
            Path to CSV file containing all concepts and synonyms.
        medcat_config (medcat.config.Config):
            MedCAT configuration file.
    Returns:
        medcat.cdb.CDB:
            MedCAT concept database containing list of entities and synonyms, without context embeddings.
    """
    logger.info('Creating concept database from concept table')
    cdb_maker = CDBMaker(config=medcat_config)
    cdb = cdb_maker.prepare_csvs([str(concept_csv_file)], full_build=True)
    return cdb


def create_vocab(cdb, training_data_list, medcat_config, output_dir, unigram_table_size):
    """Create vocabulary for word embeddings and spell check from list of training documents and CDB.

    Args:
        cdb (medcat.cdb.CDB):
            MedCAT concept database containing list of entities and synonyms.
        training_data_list (list):
            List of example documents.
        medcat_config (medcat.config.Config):
            MedCAT configuration file.
        output_dir (pathlib.Path):
            Output directory to write vocabulary and data.txt (required to create vocabulary) to.
        unigram_table_size (int):
            Size of unigram table to be initialized before creating vocabulary.

    Returns:
        medcat.vocab.Vocab:
            MedCAT vocabulary created from CDB and training documents.
    """
    logger.info('Creating and saving vocabulary')
    make_vocab = MakeVocab(cdb=cdb, config=medcat_config)
    make_vocab.make(training_data_list, out_folder=str(output_dir))
    make_vocab.add_vectors(in_path=str(output_dir/'data.txt'), unigram_table_size=unigram_table_size)
    vocab = make_vocab.vocab
    return vocab


def train_unsupervised(cdb, vocab, config, output_dir, training_data_list):
    """Perform unsupervised training and save updated CDB.

    Although not returned explicitly in this function, the CDB will be updated with context embeddings.

    Args:
        cdb (medcat.cdb.CDB):
            MedCAT concept database containing list of entities and synonyms.
        vocab (medcat.vocab.Vocab):
            MedCAT vocabulary created from CDB and training documents.
        config (medcat.config.Config):
            MedCAT configuration file.
        output_dir (pathlib.Path):
            Output directory to write updated CDB to.
        training_data_list (list):
            List of example documents.

    Returns:
        medcat.cdb.CDB:
            MedCAT concept database containing list of entities and synonyms, as well as context embeddings.
    """
    # Create MedCAT pipeline
    cat = CAT(cdb=cdb, vocab=vocab, config=config)

    # Perform unsupervised training and add model to concept database
    logger.info('Performing unsupervised training')
    cat.train(training_data_list)

    # Save output
    logger.info('Saving updated concept database')
    cdb.save(str(output_dir / 'cdb.dat'))

    return cdb


def create_models(config_file):
    """Create MedCAT CDB and Vocabulary models.

    Args:
        config_file (pathlib.Path):
            Location of model creator configuration file to specify input, output and MedCAT configuration.

    Returns:
        medcat.cat.CAT:
            Containing CDB, Vocab and Config.
    """

    # Load model creator configuration
    with open(config_file, 'r') as stream:
        creator_config = yaml.safe_load(stream)

    # Load data for unsupervised training
    with open(Path(creator_config['unsupervised_training_data_file']), 'r', encoding='utf-8') as training_data:
        training_data_list = [line.strip() for line in training_data]

    # Load MedCAT configuration
    config = Config()
    if 'medcat_config_file' in creator_config:
        config.parse_config_file(str(Path(creator_config['medcat_config_file'])))

    # Create output dir if it does not exist
    output_dir = Path(creator_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and save models
    cdb = create_cdb(Path(creator_config['concept_csv_file']), config)
    vocab = create_vocab(cdb, training_data_list, config, output_dir,
                         creator_config.get('unigram_table_size', DEFAULT_UNIGRAM_TABLE_SIZE))
    cdb = train_unsupervised(cdb, vocab, config, output_dir, training_data_list)
    cat = CAT(cdb=cdb, vocab=vocab, config=config)
    return cat


def main(config_file):
    # Setup logging
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return create_models(config_file)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML formatted file containing the parameters for model creator. An '
                                            'example can be found in `tests/model_creator/config_example.yml`',
                        type=Path)
    args = parser.parse_args()
    main(args.config_file)
