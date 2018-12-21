from spacy.tokens import Span
from cat.cat_ann import CatAnn
import numpy as np
import operator
from gensim.matutils import unitvec

CNTX_SPAN = 3
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

        self.ndone = 0

        if self.vocab is None:
            self.vocab = self.umls.vocab


    def _get_doc_words(self, doc, tkns):
        words = []
        ind = tkns[0].i
        for i in range(2*CNTX_SPAN):
            ind = ind - 1
            if ind <= 0 or len(words) == CNTX_SPAN:
                break

            if self.doc_words[ind] != 'SKIP':
                words = [self.doc_words[ind]] + words

        ind = tkns[-1].i
        cnt = 0
        for tkn in tkns:
            text = tkn._.norm
            if text in self.vocab and self.vocab.vec(text) is not None:
                words.append(text)
                cnt += 1

        limit = CNTX_SPAN*2 + cnt
        for i in range(2 * CNTX_SPAN):
            ind = ind + 1
            if ind >= len(self.doc_words) or len(words) >= limit:
                break

            if self.doc_words[ind] != 'SKIP':
                pass
                words.append(self.doc_words[ind])

        return words

    def _calc_acc(self, cui, doc, tkns):
        #start = max(0, tkns[0].i - CNTX_SPAN)
        #stop = min(len(doc), tkns[-1].i + CNTX_SPAN + 1)

        # Start from left until we find enough words or reach 0, or double the cntx_span

        cntx = None
        cnt = 0
        words = self._get_doc_words(doc, tkns)
        for word in words:
            if cntx is not None:
                cntx = cntx + self.vocab.vec(word)
            else:
                cntx = self.vocab.vec(word)
            cnt += 1

        if cui in self.umls.cui2context_vec and cnt >= (CNTX_SPAN - 1):
            print(words)
            return np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui]))
        else:
            return -1

        """
        for word in self.doc_words[start:stop]:
            if word.isnumeric() or word == 'numnum':
                word = NUM
            elif word not in self.vocab or self.vocab.vec(word) is None:
                # Skip if word is not in vocab
                continue

            if cntx is not None:
                cntx = cntx + self.vocab.vec(word)
            else:
                cntx = self.vocab.vec(word)

            cnt += 1

        if cui in self.umls.cui2context_vec and cnt >= (CNTX_SPAN - 1):
            print(self.doc_words[start:stop])
            return np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui]))
        else:
            return -1
        """
    def _add_cntx_vec(self, cui, doc, tkns):
        #start = max(0, tkns[0].i - CNTX_SPAN)
        #stop = min(len(doc), tkns[-1].i + CNTX_SPAN + 1)
        words = self._get_doc_words(doc, tkns)
        # Add cntx words
        if self.train:
            #self.umls.add_context_words(cui, self.doc_words[start:stop])
            self.umls.add_context_words(cui, words)

        cntx_vecs = []
        for word in words:
            cntx_vecs.append(self.vocab.vec(word))
        """
        for word in self.doc_words[start:stop]:
            if word.isnumeric() or word == 'numnum':
                word = NUM
            elif word not in self.vocab or self.vocab.vec(word) is None:
                # Skip if word is not in vocab or no vector present
                continue
            cntx_vecs.append(self.vocab.vec(word))
        """
        cntx = np.average(cntx_vecs, axis=0)
        if cui in self.umls.cui2context_vec and len(cntx_vecs) >= (CNTX_SPAN - 1):
            if np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui])) < 0.01:
                print("SIMILARITY::::::::::::::::::::")
                #print(self.doc_words[start:stop])
                print(words)
                print(cui)
                print(tkns)
                print(np.dot(unitvec(cntx), unitvec(self.umls.cui2context_vec[cui])))
                print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUU\n")

        if len(cntx_vecs) >= (CNTX_SPAN - 1):
            sim = self.umls.add_context_vec(cui, cntx)
            n = len(cntx_vecs)

            negs = self.vocab.get_negative_samples(n=2)
            neg_cntx_vecs = [self.vocab.vec(self.vocab.index2word[x]) for x in negs]
            neg_cntx = np.average(neg_cntx_vecs, axis=0)
            self.umls.add_context_vec(cui, neg_cntx, negative=True)


    def _add_ann(self, cui, doc, tkns, acc=-1):
        lbl = doc.vocab.strings.add(cui)
        ent = Span(doc, tkns[0].i, tkns[-1].i + 1, label=lbl)

        # If accuracy was not calculated, do it now
        if not self.train:
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
        doc_words = [t._.norm if not t._.is_punct else "SKIP" for t in doc]
        self.doc_words = [t if not t.isnumeric() and t != 'numnum' else NUM
                          for t in doc_words]
        self.doc_words = [t if t in self.vocab and self.vocab.vec(t) is not None
                          else "SKIP" for t in self.doc_words]

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


        #TODO: Disambiguate
        if False or self.adv_disambig:
            print(len(to_disamb))
            print(to_disamb)
            for t in to_disamb:
                tkns = t[0]
                name = t[1]
                tr = [tr.lower_ for tr in tkns]
                cuis = list(self.umls.name2cui[name])

                if name == "po":
                    continue

                accs = []
                cnts = []
                for cui in cuis:
                    print("&"*100)
                    self.train = False
                    acc = self._calc_acc(cui, tkns[0].doc, tkns)
                    self.train = True

                    print(cui)
                    print(name)
                    print(tkns)
                    print(acc)

                    accs.append(acc)
                    cnts.append(self.umls.cui_count.get(cui, 0))
                if self.train:
                    self.train = False
                    accs = [self._calc_acc(cui, tkns[0].doc, tkns) for cui in cuis]
                    self.train = True

                if any([x > 100 for x in cnts]):
                    # Remove accs where cnt < 100
                    for i in range(len(accs)):
                        if cnts[i] < 100:
                            accs[i] = -1
                elif all([x < 100 for x in cnts]):
                    continue

                ind = np.argmax(accs)
                cui = cuis[ind]
                acc = accs[ind]
                # Add only if acc > 0 and not training, we can't have train and adv_disambig
                if acc > 0.1:
                    if self.train:
                        # Dsable train for this ann
                        self.train = False
                        self._add_ann(cui, tkns[0].doc, tkns, acc)
                        self.train = True
                    else:
                        self._add_ann(cui, tkns[0].doc, tkns, acc)

        # Add coocurances always
        self.umls.add_coos(self._cuis)

        self.create_main_ann(doc)
        self.ndone += 1
        return doc
