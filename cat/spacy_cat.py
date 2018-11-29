from spacy.tokens import Span
import operator

class SpacyCat(object):
    def __init__(self, umls, vocab=None):
        self.umls = umls
        self.vocab = vocab
        if self.vocab is None:
            self.vocab = self.umls.vocab

    def _add_ann(self, cui, doc, tkns):
        lbl = doc.vocab.strings.add(cui)
        ent = Span(doc, tkns[0].i, tkns[-1].i + 1, label=lbl)
        doc._.ents.append(ent)


    def _scores_words(self, name, doc):
        scores = {}
        doc_words = [x._.norm for x in doc]

        for cui in self.umls.name2cui[name]:
            score = 0
            n = 0
            for word in self.umls.cui2words[cui].keys():
                if word in doc_words:
                    n += 1
                    score += self.umls.cui2words[cui][word] / self.umls.vocab[word]
            if n > 0:
                score = score / n
            scores[cui] = score
        return scores

    def _n_words_appearing(self, name, doc):
        cui = list(self.umls.name2cui[name])[0]
        doc_words = [x._.norm for x in doc]

        n = 0
        for word in self.umls.cui2words[cui].keys():
            n += doc_words.count(word)

        return n




    def add_ann(self, name, tkns, doc, to_disamb):
        one_tkn_upper = False
        if len(tkns) == 1 and tkns[0].is_upper:
            one_tkn_upper = True

        if len(name) < 5 and len(tkns) > 1:
            # Don't allow concatenation if len(name) < 5
            pass
        elif len(self.umls.name2cui[name]) == 1 and (len(name) > 3 or one_tkn_upper):
            #TODO: can't be only one tkn upper
            if len(name) > 6 or one_tkn_upper:
                cui = list(self.umls.name2cui[name])[0]
                self._add_ann(cui, doc, tkns)
            else:
                n_words = self._n_words_appearing(name, doc)
                if n_words > 2:
                    cui = list(self.umls.name2cui[name])[0]
                    self._add_ann(cui, doc, tkns)
        elif len(self.umls.name2cui[name]) > 1 and len(name) > 5:
            # Probably everything is fine, just multiple concepts with exact name
            scores = self._scores_words(name, doc)
            cui = max(scores.items(), key=operator.itemgetter(1))[0]
            self._add_ann(cui, doc, tkns)
        else:
            #print(name)
            #print(self.umls.name2cui[name])
            #print("____________")
            to_disamb.append((tkns, name))


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
        doc._.ents = []

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
                self.add_ann(name, tkns, doc, to_disamb)

            for j in range(i+1, len(_doc)):
                name = name + _doc[j]._.norm
                tkns.append(_doc[j])

                if name not in self.umls.sname2name:
                    # There is not one entity containing this words
                    break
                else:
                    if name in self.umls.name2cui:
                        self.add_ann(name, tkns, doc, to_disamb)

        self.create_main_ann(doc)

        #TODO: Disambiguate
        print(len(to_disamb))

        return doc
