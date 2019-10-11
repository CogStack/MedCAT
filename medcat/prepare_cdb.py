""" Prepartion classes for UMLS data in csv or other formats
"""

import pandas
import spacy
from spacy.tokenizer import Tokenizer
from medcat.cdb import CDB
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.preprocessing.cleaners import spacy_tag_punct, clean_name, clean_def
from spacy.tokens import Token
from medcat.utils.spacy_pipe import SpacyPipe
#from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from functools import partial

class PrepareCDB(object):
    """ Prepares CDB data in csv format for annotations,
    after everything is done the result is in the cdb field.
    """
    SEPARATOR = ""
    NAME_SEPARATOR = "|"
    CONCEPT_LENGTH_LIMIT = 8
    SKIP_STOPWORDS = False
    VERSIONS = ['CLEAN', 'RAW']

    def __init__(self, vocab=None, pretrained_cdb=None, word_tokenizer=None):
        self.vocab = vocab
        if pretrained_cdb is None:
            self.cdb = CDB()
        else:
            self.cdb = pretrained_cdb

        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all, disable=['ner', 'parser'])
        self.nlp.add_punct_tagger(tagger=partial(spacy_tag_punct, skip_stopwords=self.SKIP_STOPWORDS))
        # Get the tokenizer
        if word_tokenizer is not None:
            self.tokenizer = word_tokenizer
        else:
            self.tokenizer = self._tok

    def _tok(self, text):
        return [text]

    def prepare_csvs(self, csv_paths, sep=',', encoding=None, escapechar=None):
        """ Compile one or multiple CSVs into an internal CDB class

        csv_paths:  an array of paths to the csv files that should be processed
        sep:  if necessarya a custom separator for the csv files

        return:  Compiled CDB class
        """

        for csv_path in csv_paths:
            df = pandas.read_csv(csv_path, sep=sep, encoding=encoding, escapechar=escapechar)
            cols = list(df.columns)
            str_ind = cols.index('str')
            cui_ind = cols.index('cui')
            tui_ind = -1
            if 'tui' in cols:
                tui_ind = cols.index('tui')
            tty_ind = -1
            if 'tty' in cols:
                tty_ind = cols.index('tty')
            desc_ind = -1
            if 'desc' in cols:
                desc_ind = cols.index('desc')
            onto_ind = -1
            if 'onto' in cols:
                onto_ind = cols.index('onto')
            is_unique_ind = -1
            if 'is_unique' in cols:
                is_unique_ind = cols.index('is_unique')
            examples_ind = -1
            if 'examples' in cols:
                examples_ind = cols.index('examples')

            for ind in range(len(df)):
                names = str(df.iat[ind, str_ind]).split(self.NAME_SEPARATOR)
                for _name in names:
                    if ind % 10000 == 0:
                        print("Done: {}".format(ind))

                    skip_raw = False
                    for version in self.VERSIONS:
                        if version == "RAW" and skip_raw:
                            continue

                        # Get the cui
                        cui = str(df.iat[ind, cui_ind])

                        # Save originals
                        pretty_name = _name
                        original_name = _name
                        name = clean_name(_name)

                        # Clean and preprocess the name
                        sc_name = self.nlp(name)
                        if version == 'CLEAN':
                            tokens = [str(t.lemma_).lower() for t in sc_name if not t._.is_punct
                                      and not t._.to_skip]
                        elif version == 'RAW':
                            tokens= [str(t.lower_) for t in sc_name if not t._.is_punct
                                            and not t._.to_skip]

                        tokens_vocab = [t.lower_ for t in sc_name if not t._.is_punct]

                        # Don't allow concept names to be above concept_length_limit
                        if len(tokens) > self.CONCEPT_LENGTH_LIMIT:
                            continue

                        name = self.SEPARATOR.join(tokens)
                        tmp_name = "".join(tokens)

                        if name == self.SEPARATOR.join(tokens_vocab):
                            # Both names are the same, skip raw version
                            skip_raw = True

                        is_pref_name = False
                        if 'tty' in df.columns:
                            _tmp = str(df.iat[ind, tty_ind])
                            if _tmp.lower().strip() == 'pn':
                                is_pref_name = True

                        # Skip concepts are digits or each token is a single letter
                        length_one = [True if len(x) < 2 else False for x in tokens]
                        if tmp_name.isdigit() or all(length_one):
                            continue

                        # Create snames of the name
                        snames = []
                        sname = ""
                        for token in tokens:
                            sname = sname + token + self.SEPARATOR
                            snames.append(sname.strip())

                        # Check is unique 
                        is_unique = None
                        if 'is_unique' in df.columns:
                            _tmp = str(df.iat[ind, is_unique_ind]).strip()
                            if _tmp.lower().strip() == '0':
                                is_unique = False
                            elif _tmp.lower().strip() == '1':
                                is_unique = True

                        # Get the ontology: 'sab' in umls
                        onto = 'default'
                        if 'onto' in df.columns:
                            # Get the ontology 
                            onto = str(df.iat[ind, onto_ind])

                        # Get the tui 
                        tui = None
                        if 'tui' in df.columns:
                            _tui = str(df.iat[ind, tui_ind]).strip()
                            if len(_tui) > 0 and _tui != "nan":
                                tui = _tui
                                #TODO: If there are multiple tuis just take the first one
                                if len(tui.split(',')) > 1:
                                    tui = tui.split(',')[0]


                        # Get the concept description
                        desc = None
                        if 'desc' in df.columns:
                            _desc = str(df.iat[ind, desc_ind]).strip()
                            if len(_desc) > 0:
                                desc = _desc


                        # Add the concept
                        self.cdb.add_concept(cui, name, onto, tokens, snames,
                                tui=tui, pretty_name=pretty_name,
                                tokens_vocab=tokens_vocab, is_unique=is_unique,
                                desc=desc, original_name=original_name,
                                is_pref_name=is_pref_name)

                        # Process examples if we have them
                        examples = []
                        if 'examples' in df.columns:
                            tmp = str(df.iat[ind, examples_ind]).strip().split(self.NAME_SEPARATOR)
                            for example in tmp:
                                example = example.strip()
                                if len(example) > 0:
                                    examples.append(example)
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
                                self.cdb.add_context_vec(cui, cntx, cntx_type='MED')
        return self.cdb
