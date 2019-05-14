from spacy.tokens import Span
#from medcat.cat_ann import CatAnn
from medcat.basic_cat_ann import CatAnn
import numpy as np
import operator
from medcat.utils.loggers import basic_logger
#from gensim.matutils import unitvec
from medcat.utils.matutils import unitvec
#from pytorch_pretrained_bert import BertTokenizer
import os

log = basic_logger("spacycat")

class SpacyCat(object):
    """ A Spacy pipe module, can be easily added into a spacey pipline

    cdb:  the cdb object of class cat.cdb representing the concepts
    vocab:  vocab object of class cat.utils.vocab with vector representations
    train:  should the training be performed or not, if training is False
            the disambiguation using vectors will be performed. While training is True
            it will not be performed
    """
    DEBUG = os.getenv('DEBUG', "false").lower() == 'true'
    CNTX_SPAN = int(os.getenv('CNTX_SPAN', 6))
    CNTX_SPAN_SHORT = int(os.getenv('CNTX_SPAN_SHORT', 2))
    CNTX_SPAN_LONG = int(os.getenv('CNTX_SPAN_LONG', 0))
    MIN_CUI_COUNT = int(os.getenv('MIN_CUI_COUNT', 100))
    MIN_CUI_COUNT_STRICT = int(os.getenv('MIN_CUI_COUNT_STRICT', 15))
    # Just to be sure
    MIN_CUI_COUNT = max(MIN_CUI_COUNT_STRICT, MIN_CUI_COUNT)

    MIN_ACC = float(os.getenv('MIN_ACC', 0.12))
    MIN_CONCEPT_LENGTH = int(os.getenv('MIN_CONCEPT_LENGTH', 0))
    NEG_PROB = float(os.getenv('NEG_PROB', 0.20))
    LBL_STYLE = os.getenv('LBL_STYLE', 'long').lower()

    def __init__(self, cdb, vocab=None, train=False, force_train=False, tokenizer=None):
        self.cdb = cdb
        self.vocab = vocab
        self.train = train
        self.cat_ann = CatAnn(self.cdb, self)
        self._train_skip_names = {}
        self.force_train = force_train

        if self.vocab is None:
            self.vocab = self.cdb.vocab

        if tokenizer is None:
            self.tokenizer = self._tok
        else:
            self.tokenizer = tokenizer


    def _tok(self, text):
        return [text]


    def _get_doc_words(self, doc, tkns, span, skip_current=False, skip_words=False):
        """ Get words around a certain token

        doc:  spacy document
        tkns:  tokens around which we want to find the words
        span:  window size
        skip_current:  if True found words will not include the current tkns
        skip_words:  If True stopwords and punct will be skipped
        """
        words = []

        # Go left
        i = tkns[0].i - 1
        n = 0
        while(n < span and i >= 0):
            word = doc[i]
            if skip_words:
                if not word._.to_skip and not word.is_digit:
                    words = self.tokenizer(word._.norm) + words
                    n += 1
            else:
                words = self.tokenizer(word._.norm) + words
                n += 1
            i = i - 1

        # Add tokens if not skip_current
        if not skip_current:
            for tkn in tkns:
                words.append(tkn._.norm)

        # Go right
        i = tkns[-1].i + 1
        n = 0
        while(n < span and i < len(doc)):
            word = doc[i]
            if skip_words:
                if not word._.to_skip and not word.is_digit:
                    words = words + self.tokenizer(word._.norm)
                    n += 1
            else:
                words = words + self.tokenizer(word._.norm)
                n += 1
            i = i + 1

        return words


    def _calc_acc(self, cui, doc, tkns, name=None):
        """ Calculate the accuracy for an annotation

        cui:  concept id
        doc:  spacy document
        tkns:  tokens for the concept that was found
        name:  concept name
        """
        cntx = None
        cntx_short = None
        words = self._get_doc_words(doc, tkns, span=self.CNTX_SPAN, skip_words=True, skip_current=False)
        words_short = self._get_doc_words(doc, tkns, span=self.CNTX_SPAN_SHORT, skip_current=True)

        cntx_vecs = []
        for word in words:
            if word in self.vocab and self.vocab.vec(word) is not None:
                cntx_vecs.append(self.vocab.vec(word))

        cntx_vecs_short = []
        for word in words_short:
            if word in self.vocab and self.vocab.vec(word) is not None:
                cntx_vecs_short.append(self.vocab.vec(word))

        if len(cntx_vecs_short) > 0:
            cntx_short = np.average(cntx_vecs_short, axis=0)

        if len(cntx_vecs) > 0:
            cntx = np.average(cntx_vecs, axis=0)

        #### DEBUG ONLY ####
        if self.DEBUG:
            if cui in self.cdb.cui2context_vec and len(cntx_vecs) > 0:
                log.debug("SIMILARITY LONG::::::::::::::::::::")
                log.debug(words)
                log.debug(cui)
                log.debug(tkns)
                log.debug(np.dot(unitvec(cntx),
                          unitvec(self.cdb.cui2context_vec[cui])))
                log.debug(":::::::::::::::::::::::::::::::::::\n")

            if cui in self.cdb.cui2context_vec_short and len(cntx_vecs_short) > 0:
                log.debug("SIMILARITY SHORT::::::::::::::::::::")
                log.debug(words_short)
                log.debug(cui)
                log.debug(tkns)
                log.debug(np.dot(unitvec(cntx_short),
                          unitvec(self.cdb.cui2context_vec_short[cui])))
                log.debug(":::::::::::::::::::::::::::::::::::\n")
        #### END OF DEBUG ####

        if cui in self.cdb.cui2context_vec and len(cntx_vecs) > 0:
            sim = np.dot(unitvec(cntx), unitvec(self.cdb.cui2context_vec[cui]))

            if cui in self.cdb.cui2context_vec_short and len(cntx_vecs_short) > 0:
                sim2 = np.dot(unitvec(cntx_short), unitvec(self.cdb.cui2context_vec_short[cui]))
                sim2 = max(0, sim2)
                sim = (sim + sim2) / 2
            if name is not None:
                if cui in self.cdb.cui2pref_name:
                    if name == self.cdb.cui2pref_name[cui]:
                        sim = sim + 0.3

            """ Sometimes needed
            if cui in self.cdb.cui2ncontext_vec and self.cdb.cui_count[cui] > 20:
                neg_sim = np.dot(unitvec(cntx), unitvec(self.cdb.cui2ncontext_vec[cui]))
                log.debug("++++++NEG_SIM+++++++++++++++++++: " + str(neg_sim))
                if neg_sim > 0:
                    sim = sim - neg_sim / 2
            """

            return sim
        else:
            return -1


    def add_ncntx_vec(self, cui, words):
        cntx_vecs = []
        for word in words:
            tmp = self.tokenizer(word.lower())
            for w in tmp:
                if w in self.vocab and self.vocab.vec(w) is not None:
                    cntx_vecs.append(self.vocab.vec(w))

        cntx = np.average(cntx_vecs, axis=0)

        self.cdb.add_ncontext_vec(cui, cntx)


    def _add_cntx_vec(self, cui, doc, tkns):
        """ Add context vectors for this CUI

        cui:  concept id
        doc:  spacy document where the cui was found
        tkns:  tokens that were found for this cui
        """
        # Get words around this concept
        words = self._get_doc_words(doc, tkns, span=self.CNTX_SPAN, skip_words=True, skip_current=False)
        words_short = self._get_doc_words(doc, tkns, span=self.CNTX_SPAN_SHORT, skip_current=True)
        words_long = self._get_doc_words(doc, tkns, span=self.CNTX_SPAN_LONG, skip_current=True)

        cntx_vecs = []
        for word in words:
            if word in self.vocab and self.vocab.vec(word) is not None:
                cntx_vecs.append(self.vocab.vec(word))

        cntx_vecs_short = []
        for word in words_short:
            if word in self.vocab and self.vocab.vec(word) is not None:
                cntx_vecs_short.append(self.vocab.vec(word))

        cntx_vecs_long = []
        for word in words_long:
            if word in self.vocab and self.vocab.vec(word) is not None:
                cntx_vecs_long.append(self.vocab.vec(word))


        if len(cntx_vecs) > 0:
            cntx = np.average(cntx_vecs, axis=0)
            # Add context vectors only if we have some
            self.cdb.add_context_vec(cui, cntx, cntx_type='MED')

        if len(cntx_vecs_short) > 0:
            cntx_short = np.average(cntx_vecs_short, axis=0)
            # Add context vectors only if we have some
            self.cdb.add_context_vec(cui, cntx_short, cntx_type='SHORT', inc_cui_count=False)

        if len(cntx_vecs_long) > 0:
            cntx_long = np.average(cntx_vecs_long, axis=0)
            # Add context vectors only if we have some
            self.cdb.add_context_vec(cui, cntx_long, cntx_type='LONG', inc_cui_count=False)

            if np.random.rand() < self.NEG_PROB * 2:
                negs = self.vocab.get_negative_samples(n=10)
                neg_cntx_vecs = [self.vocab.vec(self.vocab.index2word[x]) for x in negs]
                neg_cntx = np.average(neg_cntx_vecs, axis=0)
                self.cdb.add_context_vec(cui, neg_cntx, negative=True, cntx_type='LONG',
                                          inc_cui_count=False)


        if np.random.rand() < self.NEG_PROB:
            negs = self.vocab.get_negative_samples(n=self.CNTX_SPAN * 2)
            neg_cntx_vecs = [self.vocab.vec(self.vocab.index2word[x]) for x in negs]
            neg_cntx = np.average(neg_cntx_vecs, axis=0)
            self.cdb.add_context_vec(cui, neg_cntx, negative=True, cntx_type='MED',
                                      inc_cui_count=False)

        #### DEBUG ONLY ####
        if self.DEBUG:
            if cui in self.cdb.cui2context_vec and len(cntx_vecs) > 0:
                if np.dot(unitvec(cntx), unitvec(self.cdb.cui2context_vec[cui])) < 0.01:
                    log.debug("SIMILARITY LONG::::::::::::::::::::")
                    log.debug(words)
                    log.debug(cui)
                    log.debug(tkns)
                    log.debug(np.dot(unitvec(cntx),
                              unitvec(self.cdb.cui2context_vec[cui])))
                    log.debug(":::::::::::::::::::::::::::::::::::\n")

            if cui in self.cdb.cui2context_vec_short and len(cntx_vecs_short) > 0:
                if np.dot(unitvec(cntx_short), unitvec(self.cdb.cui2context_vec_short[cui])) < 0.01:
                    log.debug("SIMILARITY SHORT::::::::::::::::::::")
                    log.debug(words_short)
                    log.debug(cui)
                    log.debug(tkns)
                    log.debug(np.dot(unitvec(cntx_short),
                              unitvec(self.cdb.cui2context_vec[cui])))
                    log.debug(":::::::::::::::::::::::::::::::::::\n")


    def _add_ann(self, cui, doc, tkns, acc=-1, name=None):
        """ Add annotation to a document

        cui:  concept id
        doc:  spacy document where the concept was found
        tkns:  tokens for this cui
        acc:  accuracy for this annotation
        name:  concept name
        """
        if self.LBL_STYLE == 'long':
            lbl = "{} - {} - {} - {} - {:.2}".format(cui, self.cdb.cui2pretty_name.get(cui, ''),
                    self.cdb.cui2tui.get(cui, ''), self.cdb.tui2name.get(self.cdb.cui2tui.get(cui, ''), ''), float(acc))
        elif self.LBL_STYLE == 'ent':
            lbl = "{} - {:.2}".format(self.cdb.tui2name.get(self.cdb.cui2tui.get(cui, ''), ''), float(acc))
        else:
            lbl = cui
        lbl = doc.vocab.strings.add(lbl)
        ent = Span(doc, tkns[0].i, tkns[-1].i + 1, label=lbl)

        ent._.acc = acc
        ent._.cui = cui
        ent._.tui = self.cdb.cui2tui.get(cui, 'None')
        doc._.ents.append(ent)

        # Increase counter for cui_count_ext
        if cui in self.cdb.cui_count_ext:
            self.cdb.cui_count_ext[cui] += 1
        else:
            self.cdb.cui_count_ext[cui] = 1

        if self.train or self.force_train:
            self._add_cntx_vec(cui, doc, tkns)

        self._cuis.append(cui)


    def _create_main_ann(self, doc):
        """ Only used for testing. Creates annotation in the spacy ents list
        from all the annotations for this document.

        doc:  spacy document
        """
        doc._.ents.sort(key=lambda x: len(x.text), reverse=True)

        tkns_in = []
        for ent in doc._.ents:
            to_add = True
            for tkn in ent:
                if tkn in tkns_in:
                    to_add = False
            if to_add:
                for tkn in ent:
                    tkns_in.append(tkn)
                doc.ents = list(doc.ents) + [ent]


    def __call__(self, doc):
        """ Annotate a document

        doc:  spacy document
        """
        self._cuis = []
        doc._.ents = []
        # Get the words in this document that should not be skipped
        doc_words = [t._.norm for t in doc if not t._.to_skip]

        _doc = []
        for token in doc:
            if not token._.to_skip:
                _doc.append(token)

        self.to_disamb = []
        for i in range(len(_doc)):
            # Go through all the tokens in this document and annotate them
            tkns = [_doc[i]]
            name = _doc[i]._.norm

            if name in self.cdb.name2cui and len(name) > self.MIN_CONCEPT_LENGTH:
                # Add annotation
                if not self.train or not self._train_skip(name) or self.force_train:
                    self.cat_ann.add_ann(name, tkns, doc, self.to_disamb, doc_words)

            for j in range(i+1, len(_doc)):
                if _doc[j]._.to_skip:
                    continue

                name = name + _doc[j]._.norm
                tkns.append(_doc[j])

                if name not in self.cdb.sname2name:
                    # There is not one entity containing this words
                    break
                else:
                    if name in self.cdb.name2cui and len(name) > self.MIN_CONCEPT_LENGTH:
                        if not self.train or not self._train_skip(name) or self.force_train:
                            self.cat_ann.add_ann(name, tkns, doc, self.to_disamb, doc_words)


        if not self.train or self.force_train:
            self.disambiguate(self.to_disamb)

        # Add coocurances
        if not self.train:
            self.cdb.add_coos(self._cuis)

        #TODO
        if self.DEBUG or True:
            self._create_main_ann(doc)

        return doc


    def _train_skip(self, name):
        if name in self._train_skip_names:
            self._train_skip_names[name] += 1
        else:
            self._train_skip_names[name] = 1

        cnt = self._train_skip_names[name]
        if cnt < self.MIN_CUI_COUNT:
            return False

        prob = 1 / (cnt / 10)
        if np.random.rand() < prob:
            return False
        else:
            return True


    def disambiguate(self, to_disamb):
        # Do vector disambiguation only if not training
        log.debug("?"*100)
        log.debug("There are {} concepts to be disambiguated.".format(len(to_disamb)))
        log.debug("The concepts are: " + str(to_disamb))

        _min_acc = self.MIN_ACC
        if self.force_train:
            _min_acc = 0.4

        for concept in to_disamb:
            # Loop over all concepts to be disambiguated
            tkns = concept[0]
            name = concept[1]
            cuis = list(self.cdb.name2cui[name])

            accs = []
            self.MIN_COUNT = self.MIN_CUI_COUNT_STRICT
            for cui in cuis:
                if self.cdb.cui_count.get(cui, 0) >= self.MIN_CUI_COUNT:
                    self.MIN_COUNT = self.MIN_CUI_COUNT
            for cui in cuis:
                # Each concept can have one or more cuis assigned
                if self.cdb.cui_count.get(cui, 0) >= self.MIN_COUNT:
                    # If this cui appeared enough times
                    accs.append(self._calc_acc(cui, tkns[0].doc, tkns, name))
                else:
                    # If not just set the accuracy to -1
                    accs.append(-1)

            ind = np.argmax(accs)
            cui = cuis[ind]
            acc = accs[ind]
            # Add only if acc > _min_acc 
            if acc > _min_acc:
                self._add_ann(cui, tkns[0].doc, tkns, acc)
