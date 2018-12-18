
    def add_ann(self, name, tkns, doc, to_disamb):
        one_tkn_upper = False
        if len(tkns) == 1 and tkns[0].is_upper:
            one_tkn_upper = True

        if len(name) < 5 and len(tkns) > 1:
            # Don't allow concatenation of tokens if len(name) < 5
            pass
        elif len(self.umls.name2cui[name]) == 1 and (len(name) > 3
                or (len(name) > 2 and one_tkn_upper)):
            # Disambig type 2. Concept is detected, but we are not sure
            #is it really that concept.
            if len(name) > 6 or one_tkn_upper:
                # No disambiguation if it is all upper or length if > 6
                cui = list(self.umls.name2cui[name])[0]
                self._add_ann(cui, doc, tkns, acc=1)
            else:
                # Disambiguation is needed, but simple approach. Just
                #check the number of words
                n_words = self._n_words_appearing(name, doc)
                cui = list(self.umls.name2cui[name])[0]
                d = self.umls.cui2words[cui]
                perc =  0
                cnt = 0
                if name in d:
                    perc = d[name] / sum(d.values())
                    cnt = d[name]
                if n_words > len(tkns) or (len(name) > 2 and (perc > 0.2 or cnt > 2)):
                    # We want other words describing this concept to appear also, not
                    #just the ones that were found
                    self._add_ann(cui, doc, tkns, acc=1)
                else:
                    to_disamb.append((list(tkns), name))

        elif len(self.umls.name2cui[name]) > 1 and (len(name) > 5 or
                 (len(name) > 2 and one_tkn_upper)):
            # Probably everything is fine, just multiple concepts with exact name
            #TODO: add disambig using vectors
            scores = self._scores_words(name, doc)
            print("-"*150)
            print(scores)
            acc = self.softmax(scores.values())
            if acc > 0.6:
                cui = max(scores.items(), key=operator.itemgetter(1))[0]
                self._add_ann(cui, doc, tkns, acc=acc)
            else:
                print(tkns)
                print(name)
                print("******************************")

                to_disamb.append((list(tkns), name))
        else:
            if not one_tkn_upper and name in self.umls.stopwords:
                # Skip
                pass
            else:
                #print(name)
                #print(self.umls.name2cui[name])
                to_disamb.append((list(tkns), name))
