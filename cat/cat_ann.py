import numpy as np
import operator

class CatAnn(object):
    def __init__(self, umls, _add_ann):
        self.umls = umls
        self._add_ann = _add_ann


    def add_ann(self, name, tkns, doc, to_disamb, doc_words):
        one_tkn_upper = False
        name_case = True
        if len(tkns) == 1 and tkns[0].is_upper:
            one_tkn_upper = True
        for tkn in tkns:
            if not tkn.is_upper:
                name_case = False

        # Don't allow concatenation of tokens if len(name) < 5
        if not(len(name) < 5 and len(tkns) > 1):
            # Name must have > 2, if not disambiguation is a must
            if len(name) > 3:
                if len(self.umls.name2cui[name]) == 1:
                    cui = list(self.umls.name2cui[name])[0]
                    # Disambiguation needed if length of string < 6
                    if len(name) < 6:
                        # Case must agree or first can be lower_case
                        if not name_case or self.umls.name_isupper[name] == name_case:
                            """ Try not using this, disambiguate everything
                            if name_case:
                                # Means match is upper in both cases, we can tag
                                self._add_ann(cui, doc, tkns, acc=1)
                            """
                            if not name_case or (len(name) > 4):
                                # Means name is not upper, disambiguation is needed
                                n_words, words_cnt = self._n_words_appearing(name, doc, doc_words)
                                d = self.umls.cui2words[cui]
                                perc =  0
                                cnt = 0
                                if name in d:
                                    perc = d[name] / sum(d.values())
                                    cnt = d[name]
                                if (n_words > len(tkns) and words_cnt > 5) or (perc > 0.2 or cnt > 5):
                                    self._add_ann(cui, doc, tkns, acc=1)
                                else:
                                    to_disamb.append((list(tkns), name))
                            else:
                                to_disamb.append((list(tkns), name))
                        else:
                            # Case dosn't match add to to_disamb
                            to_disamb.append((list(tkns), name))
                    else:
                        # Longer than 5 letters, just add concept
                        cui = list(self.umls.name2cui[name])[0]
                        self._add_ann(cui, doc, tkns, acc=1)
                else:
                    # Means we have more than one cui for this name
                    scores = self._scores_words(name, doc, doc_words)
                    #print("-"*150)
                    #print(scores)
                    #print(tkns)
                    acc = self.softmax(scores.values())
                    if len(name) < 6:
                        if self.umls.name_isupper[name] == name_case or (not name_case and len(name) > 3):
                            # Means match is upper in both cases, tag if acc > 0.6
                            if acc > 0.6:
                                cui = max(scores.items(), key=operator.itemgetter(1))[0]
                                self._add_ann(cui, doc, tkns, acc=acc)
                            else:
                                to_disamb.append((list(tkns), name))
                        else:
                            to_disamb.append((list(tkns), name))
                    else:
                        # We can be almost sure that everything is fine, threshold of 0.2
                        if acc > 0.3:
                            cui = max(scores.items(), key=operator.itemgetter(1))[0]
                            self._add_ann(cui, doc, tkns, acc=acc)
                        else:
                            to_disamb.append((list(tkns), name))
                    #print("*"*150)
            else:
                if name not in self.umls.stopwords: # and one_tkn_upper:
                    # Only if abreviation and not in stopwords
                    to_disamb.append((list(tkns), name))


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        x = list(x)
        if not any(x):
            return 0
        e_x = np.exp(x / np.max(x))
        #print(e_x / e_x.sum())
        return max(e_x / e_x.sum())


    def _scores_words(self, name, doc, doc_words):
        scores = {}

        name_cnt = self.umls.name2cnt[name]
        sm = sum(name_cnt.values())

        for cui in self.umls.name2cui[name]:
            score = 0
            n = 0
            for word in self.umls.cui2words[cui].keys():
                if word in doc_words:
                    n += 1
                    score += self.umls.cui2words[cui][word] / self.umls.vocab[word]
            if n > 0:
                score = score / n

            # Add proportion for name count
            score = (score + (name_cnt[cui] / sm)) / 2

            # Check is this the prefered name for this concept
            if len(name) > 3:
                if cui in self.umls.cui2pref_name:
                    if name == self.umls.cui2pref_name[cui]:
                        score = score * 2
            scores[cui] = score
        return scores


    def _n_words_appearing(self, name, doc, doc_words):
        cui = list(self.umls.name2cui[name])[0]

        n = 0
        cnt = 0
        for word in self.umls.cui2words[cui].keys():
            if word in doc_words:
                n += 1
                cnt += doc_words.count(word)
        return n, cnt
