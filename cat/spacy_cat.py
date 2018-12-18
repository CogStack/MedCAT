from spacy.tokens import Span
from cat.cat_ann import CatAnn
import numpy as np
import operator
from gensim.matutils import unitvec

CNTX_SPAN = 6
NUM = "NUMNUM"

class SpacyCat(object):
    """

    """
    def __init__(self, umls, vocab=None, train=False, adv_disambig=False):
        self.umls = umls
        self.vocab = vocab
        self.train = train
        self.adv_disambig = adv_disambig
        self.cat_ann = CatAnn(self.umls, self._add_ann)

        if self.vocab is None:
            self.vocab = self.umls.vocab


    def _calc_acc(self, cui, doc, tkns):
        start = max(0, tkns[0].i - CNTX_SPAN)
        stop = min(len(doc), tkns[-1].i + CNTX_SPAN + 1)
        # Add cntx words
        self.umls.add_context_words(cui, self.doc_words[start:stop])

        cntx = None
        cnt = 0
        for word in self.doc_words[start:stop]:
            if word.isnumeric() or word == 'numnum':
                word = NUM
            elif word not in self.vocab:
                # Skip if word is not in vocab
                continue

            if cntx is not None:
                cntx = cntx + self.vocab.vec(word)
            else:
                cntx = self.vocab.vec(word)

            cnt += 1

        if cui in self.umls.cui2context_vec and cnt > 4:
            print(self.doc_words[start:stop])
            return np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui]))
        else:
            return -1

    def _add_cntx_vec(self, cui, doc, tkns):
        start = max(0, tkns[0].i - CNTX_SPAN)
        stop = min(len(doc), tkns[-1].i + CNTX_SPAN + 1)
        # Add cntx words
        self.umls.add_context_words(cui, self.doc_words[start:stop])

        cntx_vecs = []
        for word in self.doc_words[start:stop]:
            if word.isnumeric() or word == 'numnum':
                word = NUM
            elif word not in self.vocab or self.vocab.vec(word) is None:
                # Skip if word is not in vocab or no vector present
                continue
            cntx_vecs.append(self.vocab.vec(word))

        cntx = np.average(cntx_vecs, axis=0)
        if cui in self.umls.cui2context_vec and len(cntx_vecs) > 4:
            if np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui])) < 0.01:
                print("SIMILARITY::::::::::::::::::::")
                print(self.doc_words[start:stop])
                print(cui)
                print(tkns)
                print(np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui])))
                print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\n")

        if len(cntx_vecs) > 4:
            self.umls.add_context_vec(cui, cntx)

            # Add negative words, try with 2 for now
            negs = self.vocab.get_negative_samples(n=1)
            neg_cntx_vecs = [self.vocab.vec(self.vocab.index2word[x]) for x in negs]
            neg_cntx = np.average(neg_cntx_vecs, axis=0)
            self.umls.add_context_vec(cui, neg_cntx, negative=True)


    def _add_ann(self, cui, doc, tkns, acc=-1):
        lbl = doc.vocab.strings.add(cui)
        ent = Span(doc, tkns[0].i, tkns[-1].i + 1, label=lbl)
        if acc == -1:
            # If accuracy was not calculated, do it now
            acc = self._calc_acc(cui, doc, tkns)
        ent._.acc = acc
        doc._.ents.append(ent)

        # Increase cui count for this one
        if cui in self.umls.cui_count:
            self.umls.cui_count[cui] += 1
        else:
            self.umls.cui_count[cui] = 1

        if self.train:
            self._add_cntx_vec(cui, doc, tkns)
            self._cuis.append(cui)


    def create_main_ann(self, doc):
        # Sort ents by length
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

    def add_cor_matrix(self, doc):
        pass


    def __call__(self, doc):
        self._cuis = []
        doc._.ents = []
        doc_words = [t._.norm if not t._.is_punct else "PUCT" for t in doc]
        self.doc_words = [t if not t.isnumeric() else NUM for t in doc_words]

        _doc = []
        for token in doc:
            if not token._.is_punct:
                if token._.norm in self.umls.vocab or token._.norm in self.umls.sname2name:
                    _doc.append(token)

        to_disamb = []
        for i in range(len(_doc)):
            tkns = [_doc[i]]
            name = _doc[i]._.norm

            if name in self.umls.name2cui:
                # Add annotation
                self.cat_ann.add_ann(name, tkns, doc, to_disamb, doc_words)

            for j in range(i+1, len(_doc)):
                name = name + _doc[j]._.norm
                tkns.append(_doc[j])

                if name not in self.umls.sname2name:
                    # There is not one entity containing this words
                    break
                else:
                    if name in self.umls.name2cui:
                        self.cat_ann.add_ann(name, tkns, doc, to_disamb, doc_words)

        self.create_main_ann(doc)

        #TODO: Disambiguate
        if True or self.adv_disambig:
            print(len(to_disamb))
            print(to_disamb)
            for t in to_disamb:
                tkns = t[0]
                name = t[1]
                tr = [tr.lower_ for tr in tkns]
                if "po" in list(tr):
                    continue 

                cuis = self.umls.name2cui[name]

                for cui in cuis:
                    acc = self._calc_acc(cui, tkns[0].doc, tkns)
                    # Add only if acc > 0 and not training, we can't have train and adv_disambig
                    if acc > 0 and not self.train:
                        self._add_ann(cui, tkns[0].doc, tkns, acc)
                    print("&"*100)
                    print(cui)
                    print(name)
                    print(tkns)
                    print(acc)

        # Add coocurances always
        self.umls.add_coos(self._cuis)

        return doc

