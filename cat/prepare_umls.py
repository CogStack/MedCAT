""" Prepartion classes for UMLS data in csv or other formats
"""

import pandas
import spacy
from spacy.tokenizer import Tokenizer
from cat.umls import UMLS
from cat.preprocessing.tokenizers import spacy_split_all
from cat.preprocessing.cleaners import spacy_tag_punct, clean_umls, clean_def
from spacy.tokens import Token
from cat.utils.spacy_pipe import SpacyPipe
from pytorch_pretrained_bert import BertTokenizer
import numpy as np

SEPARATOR = ""

class PrepareUMLS(object):
    """ Prepares UMLS data in csv format for annotations,
    after everything is done the result is in the umls field.
    """
    def __init__(self, vocab=None):
        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all, disable=['ner', 'parser', 'tagger'])
        self.nlp.add_punct_tagger(tagger=spacy_tag_punct)

        self.umls = UMLS()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.vocab = vocab

    def prepare_csvs(self, csv_paths, sep=',', concept_length_limit=6):
        """ Prepare one or multiple csvs

        csv_paths:  an array of paths to the csv files that should be processed
        """
        for csv_path in csv_paths:
            df = pandas.read_csv(csv_path, sep=sep)
            for ind in range(len(df)):
                names = str(df.iloc[ind]['str']).split("||")
                for _name in names:
                    if ind % 10000 == 0:
                        print("Done: {}".format(ind))
                    pretty_name = _name
                    name = clean_umls(_name)
                    # Clean and preprocess the name
                    doc = self.nlp(name)
                    tokens = [str(t.lemma_).lower() for t in doc if not t._.is_punct and not t._.to_skip]

                    # Don't allow concept names to be above concept_length_limit
                    if len(tokens) > concept_length_limit:
                        continue

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
                    for token in tokens:
                        sname = sname + token + SEPARATOR
                        snames.append(sname.strip())

                    # Check is prefered name, it is if the column "TTY" equals PN
                    is_pref_name = False
                    if 'tty' in df.columns:
                        _tmp = str(df.iloc[ind]['tty'])
                        if _tmp.lower().strip() == 'pn':
                            is_pref_name = True

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

                    desc = None
                    if 'def' in df.columns:
                        tmp = str(df.iloc[ind]['def']).strip()
                        if len(tmp) > 0:
                            desc = tmp

                    self.umls.add_concept(cui, name, onto, tokens, snames, isupper=isupper,
                            is_pref_name=is_pref_name, tui=tui, pretty_name=pretty_name, desc=desc)

                    # If we had desc we can also add vectors 
                    if desc is not None:
                        doc = self.nlp(clean_def(desc))
                        cntx = []
                        for word in doc:
                            if not word._.to_skip:
                                for w in self.tokenizer.tokenize(word.lower_):
                                    if w in self.vocab and self.vocab.vec(w) is not None:
                                        cntx.append(self.vocab.vec(w))
                        if len(cntx) > 1:
                            cntx = np.average(cntx, axis=0)
                            self.umls.add_context_vec(cui, cntx, cntx_type='LONG')
                            # Increase cui count because we added the context
                            if cui in self.umls.cui_count:
                                self.umls.cui_count[cui] += 1
                            else:
                                self.umls.cui_count[cui] = 1

        return self.umls
