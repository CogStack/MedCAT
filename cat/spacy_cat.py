from spacy.tokens import Span
from cat.cat_ann import CatAnn
import numpy as np
import operator
from cat.utils.loggers import basic_logger
#from gensim.matutils import unitvec
from cat.utils.matutils import unitvec
#from pytorch_pretrained_bert import BertTokenizer

DEBUG = False
CNTX_SPAN = 5
CNTX_SPAN_SHORT = 2
NUM = "NUMNUM"
MIN_CUI_COUNT = 500
MIN_CUI_COUNT_STRICT = 1
MIN_ACC = 0.1
MIN_CONCEPT_LENGTH = 0

log = basic_logger("spacycat")

class SpacyCat(object):
    """ A Spacy pipe module, can be easily added into a spacey pipline

    umls:  the umls object of class cat.umls representing the concepts
    vocab:  vocab object of class cat.utils.vocab with vector representations
    train:  should the training be performed or not, if training is False
            the disambiguation using vectors will be performed. While training is True
            it will not be performed
    """
    def __init__(self, umls, vocab=None, train=False, force_train=False, tokenizer=None):
        self.umls = umls
        self.vocab = vocab
        self.train = train
        self.cat_ann = CatAnn(self.umls, self)
        self._train_skip_names = {}
        self.force_train = force_train

        if self.vocab is None:
            self.vocab = self.umls.vocab

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
        words = self._get_doc_words(doc, tkns, span=CNTX_SPAN, skip_words=True)
        words_short = self._get_doc_words(doc, tkns, span=CNTX_SPAN_SHORT, skip_current=True)

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
        if DEBUG:
            if cui in self.umls.cui2context_vec and len(cntx_vecs) > 0:
                log.debug("SIMILARITY LONG::::::::::::::::::::")
                log.debug(words)
                log.debug(cui)
                log.debug(tkns)
                log.debug(np.dot(unitvec(cntx),
                          unitvec(self.umls.cui2context_vec[cui])))
                log.debug(":::::::::::::::::::::::::::::::::::\n")

            if cui in self.umls.cui2context_vec_short and len(cntx_vecs_short) > 0:
                log.debug("SIMILARITY SHORT::::::::::::::::::::")
                log.debug(words_short)
                log.debug(cui)
                log.debug(tkns)
                log.debug(np.dot(unitvec(cntx_short),
                          unitvec(self.umls.cui2context_vec_short[cui])))
                log.debug(":::::::::::::::::::::::::::::::::::\n")
        #### END OF DEBUG ####

        if cui in self.umls.cui2context_vec and len(cntx_vecs) > 0:
            sim = np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui]))

            if cui in self.umls.cui2context_vec_short and len(cntx_vecs_short) > 0:
                sim2 = np.dot(unitvec(cntx_short), unitvec(self.umls.cui2context_vec_short[cui]))
                if sim2 > 0:
                    sim = (sim + sim2) / 2
            if name is not None:
                if cui in self.umls.cui2pref_name:
                    if name == self.umls.cui2pref_name[cui]:
                        sim = sim + 0.2

            """ Just in case, it works better when there is a lot of data
            if cui in self.umls.cui2ncontext_vec and self.umls.cui_count[cui] > 20:
                neg_sim = np.dot(unitvec(cntx), unitvec(self.umls.cui2ncontext_vec[cui]))
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

        self.umls.add_ncontext_vec(cui, cntx)


    def _add_cntx_vec(self, cui, doc, tkns):
        """ Add context vectors for this CUI

        cui:  concept id
        doc:  spacy document where the cui was found
        tkns:  tokens that were found for this cui
        """
        # Get words around this concept
        words = self._get_doc_words(doc, tkns, span=CNTX_SPAN, skip_words=True)
        words_short = self._get_doc_words(doc, tkns, span=CNTX_SPAN_SHORT, skip_current=True)

        cntx_vecs = []
        for word in words:
            if word in self.vocab and self.vocab.vec(word) is not None:
                cntx_vecs.append(self.vocab.vec(word))

        cntx_vecs_short = []
        for word in words_short:
            if word in self.vocab and self.vocab.vec(word) is not None:
                cntx_vecs_short.append(self.vocab.vec(word))

        cntx = np.average(cntx_vecs, axis=0)
        cntx_short = np.average(cntx_vecs_short, axis=0)

        if len(cntx_vecs) > 0:
            # Add context vectors only if we have some
            self.umls.add_context_vec(cui, cntx, cntx_type='LONG')

        if len(cntx_vecs_short) > 0:
            # Add context vectors only if we have some
            self.umls.add_context_vec(cui, cntx_short, cntx_type='SHORT', inc_cui_count=False)

        #### DEBUG ONLY ####
        if DEBUG:
            if cui in self.umls.cui2context_vec and len(cntx_vecs) >= (CNTX_SPAN - 1):
                if np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui])) < 0.01:
                    log.debug("SIMILARITY LONG::::::::::::::::::::")
                    log.debug(words)
                    log.debug(cui)
                    log.debug(tkns)
                    log.debug(np.dot(unitvec(cntx),
                              unitvec(self.umls.cui2context_vec[cui])))
                    log.debug(":::::::::::::::::::::::::::::::::::\n")

            if cui in self.umls.cui2context_vec_short and len(cntx_vecs_short) > 0:
                if np.dot(unitvec(cntx_short), unitvec(self.umls.cui2context_vec_short[cui])) < 0.01:
                    log.debug("SIMILARITY SHORT::::::::::::::::::::")
                    log.debug(words_short)
                    log.debug(cui)
                    log.debug(tkns)
                    log.debug(np.dot(unitvec(cntx_short),
                              unitvec(self.umls.cui2context_vec[cui])))
                    log.debug(":::::::::::::::::::::::::::::::::::\n")


    def _add_ann(self, cui, doc, tkns, acc=-1, name=None):
        """ Add annotation to a document

        cui:  concept id
        doc:  spacy document where the concept was found
        tkns:  tokens for this cui
        acc:  accuracy for this annotation
        name:  concept name
        """
        #lbl = "{} - {:.2} - {}".format(cui, float(acc), self.umls.cui2pretty_name.get(cui, ''))
        lbl = cui
        lbl = doc.vocab.strings.add(lbl)
        ent = Span(doc, tkns[0].i, tkns[-1].i + 1, label=lbl)

        ent._.acc = acc
        doc._.ents.append(ent)

        # Increase counter for cui_count_ext
        if cui in self.umls.cui_count_ext:
            self.umls.cui_count_ext[cui] += 1
        else:
            self.umls.cui_count_ext[cui] = 1

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

            if name in self.umls.name2cui and len(name) > MIN_CONCEPT_LENGTH:
                # Add annotation
                if not self.train or not self._train_skip(name) or self.force_train:
                    self.cat_ann.add_ann(name, tkns, doc, self.to_disamb, doc_words)

            for j in range(i+1, len(_doc)):
                if _doc[j]._.to_skip:
                    continue

                name = name + _doc[j]._.norm
                tkns.append(_doc[j])

                if name not in self.umls.sname2name:
                    # There is not one entity containing this words
                    break
                else:
                    if name in self.umls.name2cui and len(name) > MIN_CONCEPT_LENGTH:
                        if not self.train or not self._train_skip(name) or self.force_train:
                            self.cat_ann.add_ann(name, tkns, doc, self.to_disamb, doc_words)


        if not self.train or self.force_train:
            self.disambiguate(self.to_disamb)

        # Add coocurances
        self.umls.add_coos(self._cuis)

        if DEBUG or True:
            self._create_main_ann(doc)

        return doc


    def _train_skip(self, name):
        if name in self._train_skip_names:
            self._train_skip_names[name] += 1
        else:
            self._train_skip_names[name] = 1

        cnt = self._train_skip_names[name]
        if cnt < MIN_CUI_COUNT:
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

        _min_acc = MIN_ACC
        if self.force_train:
            _min_acc = 0.4

        for concept in to_disamb:
            # Loop over all concepts to be disambiguated
            tkns = concept[0]
            name = concept[1]
            cuis = list(self.umls.name2cui[name])

            accs = []
            MIN_COUNT = MIN_CUI_COUNT_STRICT
            for cui in cuis:
                if self.umls.cui_count.get(cui, 0) >= MIN_CUI_COUNT:
                    MIN_COUNT = MIN_CUI_COUNT
            for cui in cuis:
                # Each concept can have one or more cuis assigned
                if self.umls.cui_count.get(cui, 0) >= MIN_COUNT:
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
