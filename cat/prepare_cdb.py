""" Prepartion classes for UMLS data in csv or other formats
"""

import pandas
import spacy
from spacy.tokenizer import Tokenizer
from cat.cdb import CDB
from cat.preprocessing.tokenizers import spacy_split_all
from cat.preprocessing.cleaners import spacy_tag_punct, clean_name, clean_def
from spacy.tokens import Token
from cat.utils.spacy_pipe import SpacyPipe
#from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from functools import partial

SEPARATOR = ""
CONCEPT_LENGTH_LIMIT = 6

class PrepareCDB(object):
    """ Prepares CDB data in csv format for annotations,
    after everything is done the result is in the cdb field.
    """
    def __init__(self, vocab=None, pretrained_cdb=None, word_tokenizer=None):
        self.vocab = vocab
        if pretrained_cdb is None:
            self.cdb = CDB()
        else:
            self.cdb = pretrained_cdb

        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all, disable=['ner', 'parser'])
        self.nlp.add_punct_tagger(tagger=partial(spacy_tag_punct, skip_stopwords=False))
        # Get the tokenizer
        if word_tokenizer is not None:
            self.tokenizer = word_tokenizer
        else:
            self.tokenizer = self._tok

    def _tok(self, text):
        return [text]

    def prepare_csvs(self, csv_paths, sep=','):
        """ Compile one or multiple CSVs into an internal CDB class

        csv_paths:  an array of paths to the csv files that should be processed
        sep:  if necessarya a custom separator for the csv files

        return:  Compiled CDB class
        """
        for csv_path in csv_paths:
            df = pandas.read_csv(csv_path, sep=sep)
            for ind in range(len(df)):
                names = str(df.iloc[ind]['str']).split("||")
                for _name in names:
                    if ind % 10000 == 0:
                        print("Done: {}".format(ind))
                    pretty_name = _name
                    name = clean_name(_name)
                    # Clean and preprocess the name
                    sc_name = self.nlp(name)
                    tokens = [str(t.lemma_).lower() for t in sc_name if not t._.is_punct and not t._.to_skip]

                    # Don't allow concept names to be above concept_length_limit
                    if len(tokens) > CONCEPT_LENGTH_LIMIT:
                        continue

                    name = SEPARATOR.join(tokens)
                    _name = "".join(tokens)
                    length_one = [True if len(x) < 2 else False for x in tokens]

                    # Skip concepts are digits or each token is a single letter
                    if _name.isdigit() or all(length_one):
                        continue

                    # Create snames of the name
                    snames = []
                    sname = ""
                    for token in tokens:
                        sname = sname + token + SEPARATOR
                        snames.append(sname.strip())

                    # Check is fullname
                    fullname = False
                    if 'fullname' in df.columns:
                        _tmp = str(df.iloc[ind]['fullname']).strip()
                        if _tmp.lower().strip() == '1':
                            fullname = True

                    onto = 'default'
                    if 'sab' in df.columns:
                        # Get the ontology 
                        onto = df.iloc[ind]['sab']

                    # Get the cui
                    cui = df.iloc[ind]['cui']

                    # Get the tui 
                    tui = None
                    if 'tui' in df.columns:
                        tui = str(df.iloc[ind]['tui'])
                        #TODO: If there are multiple tuis just take the first one
                        if len(tui.split(',')) > 1:
                            tui = tui.split(',')[0]

                    examples = []
                    if 'examples' in df.columns:
                        tmp = str(df.iloc[ind]['examples']).strip().split("||")
                        for example in tmp:
                            example = example.strip()
                            if len(example) > 0:
                                examples.append(example)

                    self.cdb.add_concept(cui, name, onto, tokens, snames,
                            is_pref_name=is_pref_name, tui=tui, pretty_name=pretty_name, desc=desc)

                    # If we have examples
                    for example in examples:
                        doc = self.nlp(example)
                        cntx = []
                        for word in doc:
                            if not word._.to_skip:
                                for w in self.tokenizer(word._.norm):
                                    if w in self.vocab and self.vocab.vec(w) is not None:
                                        cntx.append(self.vocab.vec(w))
                        if len(cntx) > 1:
                            cntx = np.average(cntx, axis=0)
                            self.cdb.add_context_vec(cui, cntx, cntx_type='LONG')

        return self.cdb
