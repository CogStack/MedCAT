""" Prepartion classes for UMLS data in csv or other formats
"""

import pandas
import spacy
from spacy.tokenizer import Tokenizer
from cat.umls import UMLS
from preprocessing.tokenizers import spacy_split_all
from preprocessing.cleaners import spacy_tag_punct, clean_umls
from spacy.tokens import Token
from preprocessing.spacy_pipe import SpacyPipe

SEPARATOR = ""

class PrepareUMLS(object):
    """ Prepares UMLS data in csv format for annotations,
    after everything is done the result is in the umls field.
    """
    def __init__(self):
        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all)
        self.nlp.add_punct_tagger(tagger=spacy_tag_punct)

        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.umls = UMLS(stopwords=stopwords)

    def prepare_csvs(self, csv_paths):
        """ Prepare one or multiple csvs

        csv_paths:  an array of paths to the csv files that should be processed
        """
        for csv_path in csv_paths:
            df = pandas.read_csv(csv_path)

            for ind in range(len(df)):
                if ind % 10000 == 0:
                    print("Done: {}".format(ind))

                name = clean_umls(str(df.iloc[ind]['str']))
                # Clean and preprocess the name
                doc = self.nlp(name)
                tokens = [t.lower_ for t in doc if not t._.is_punct and not t._.to_skip]
                name = SEPARATOR.join(tokens)
                _name = "".join(tokens)
                length_one = [True if len(x) < 2 else False for x in tokens]
                # Skip concepts are digits or each token is a single letter
                if _name.isdigit() or all(length_one):
                    continue

                # Create snames of the name
                snames = []
                sname = ""
                for token in [t for t in doc if not t._.is_punct]:
                    sname = sname + token.lower_ + SEPARATOR
                    snames.append(sname.strip())

                onto = df.iloc[ind]['sab']
                cui = df.iloc[ind]['cui']

                self.umls.add_concept(cui, name, onto, tokens, snames)
