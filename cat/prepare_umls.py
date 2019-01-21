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
                pretty_name = str(df.iloc[ind]['str'])
                name = clean_umls(str(df.iloc[ind]['str']))
                # Clean and preprocess the name
                doc = self.nlp(name)
                tokens = [t.lower_ for t in doc if not t._.is_punct and not t._.to_skip]
                isupper = False
                if len(doc) == 1:
                    if doc[0].is_upper and len(doc[0]) > 1:
                        isupper = True
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

                # Check is prefered name, it is if the column "TTY" equals PN
                is_pref_name = False
                if 'tty' in df.columns:
                    _tmp = str(df.iloc[ind]['tty'])
                    if _tmp.lower().strip() == 'pn':
                        is_pref_name = True

                sab = 'default'
                if 'sab' in df.columns:
                    # Get the ontology 
                    onto = df.iloc[ind]['sab']

                # Get the cui
                cui = df.iloc[ind]['cui']

                # Get the tui 
                tui = None
                if 'tui' in df.columns:
                    tui = str(df.iloc[ind]['tui'])

                self.umls.add_concept(cui, name, onto, tokens, snames, isupper=isupper,
                        is_pref_name=is_pref_name, tui=tui, pretty_name=pretty_name)
