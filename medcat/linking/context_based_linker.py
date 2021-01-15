from medcat.utils.filters import check_filters
from medcat.linking.vector_context_model import ContextModel
import random

class Linker(object):
    r''' Link to a biomedical database.

    Args:
        cdb
        vocab
        config
    '''
    def __init__(self, cdb, vocab, config):
        self.cdb = cdb
        self.vocab = vocab
        self.config = config
        self.context_model = ContextModel(self.cdb, self.vocab, self.config)
        # Counter for how often did a pair (name,cui) appear and was used during training
        self.train_counter = {}


    def _train(self, cui, entity, doc, add_negative=True):
        name = "{} - {}".format(entity._.detected_name, cui)
        if self.train_counter.get(name, 0) > self.config.linking['subsample_after']:
            if random.random() < 1 / (self.train_counter.get(name) - self.config.linking['subsample_after']):
                self.context_model.train(cui, entity, doc, negative=False)
                if add_negative and self.config.linking['negative_probability'] >= random.random():
                    self.context_model.train_using_negative_sampling(cui)
                self.train_counter[name] = self.train_counter.get(name, 0) + 1
        else:
            # Always train
            self.context_model.train(cui, entity, doc, negative=False)
            if add_negative and self.config.linking['negative_probability'] >= random.random():
                self.context_model.train_using_negative_sampling(cui)
            self.train_counter[name] = self.train_counter.get(name, 0) + 1


    def __call__(self, doc):
        r'''
        '''
        cnf_l = self.config.linking
        linked_entities = []

        doc_tkns = [tkn for tkn in doc if not tkn._.to_skip]
        doc_tkn_ids = [tkn.idx for tkn in doc_tkns]

        if cnf_l['train']:
            # Run training
            for entity in doc._.ents:
                # Check does it have a detected name
                if entity._.detected_name is not None:
                    name = entity._.detected_name
                    cuis = entity._.link_candidates

                    if len(name) >= cnf_l['disamb_length_limit']:
                        if len(cuis) == 1:
                            # N - means name must be disambiguated, is not the prefered
                            #name of the concept, links to other concepts also.
                            if self.cdb.name2cuis2status[name][cuis[0]] != 'N':
                                self._train(cui=cuis[0], entity=entity, doc=doc)
                                entity._.cui = cuis[0]
                                entity._.context_similarity = 1
                                linked_entities.append(entity)
                        else:
                            for cui in cuis:
                                if self.cdb.name2cuis2status[name][cui] in {'P', 'PD'}:
                                    self._train(cui=cui, entity=entity, doc=doc)
                                    # It should not be possible that one name is 'P' for two CUIs,
                                    #but it can happen - and we do not care.
                                    entity._.cui = cui
                                    entity._.context_similarity = 1
                                    linked_entities.append(entity)
        else:
            for entity in doc._.ents:
                # Check does it have a detected name
                if entity._.link_candidates is not None:
                    if entity._.detected_name is not None:
                        name = entity._.detected_name
                        cuis = entity._.link_candidates

                        if len(cuis) > 0:
                            do_disambiguate = False
                            if len(name) < cnf_l['disamb_length_limit']:
                                do_disambiguate = True
                            elif len(cuis) == 1 and self.cdb.name2cuis2status[name][cuis[0]] in {'N', 'PD'}:
                                # PD means it is preferred but should still be disambiguated and N is disamb always
                                do_disambiguate = True
                            elif len(cuis) > 1:
                                do_disambiguate = True

                            if do_disambiguate:
                                cui, context_similarity = self.context_model.disambiguate(cuis, entity, name, doc)
                            else:
                                cui = cuis[0]
                                if self.config.linking['always_calculate_similarity']:
                                    context_similarity = self.context_model.similarity(cui, entity, doc)
                                else:
                                    context_similarity = 1 # Direct link, no care for similarity
                    else:
                        # No name detected, just disambiguate
                        cui, context_similarity = self.context_model.disambiguate(entity._.link_candidates, entity, 'unk-unk', doc)

                    # Add the annotation if it exists and if above threshold and in filters
                    if cui and context_similarity >= self.config.linking['similarity_threshold'] and check_filters(cui, self.config.linking['filters']):
                        entity._.cui = cui
                        entity._.context_similarity = context_similarity
                        linked_entities.append(entity)

        doc._.ents = linked_entities
        self._create_main_ann(doc)
        return doc


    def _create_main_ann(self, doc, tuis=None):
        # TODO: Separate into another piece of the pipeline
        """ Creates annotation in the spacy ents list
        from all the annotations for this document.

        doc:  spacy document
        """
        doc._.ents.sort(key=lambda x: len(x.text), reverse=True)

        tkns_in = set()
        main_anns = []
        for ent in doc._.ents:
            if tuis is None or ent._.tui in tuis:
                to_add = True
                for tkn in ent:
                    if tkn in tkns_in:
                        to_add = False
                if to_add:
                    for tkn in ent:
                        tkns_in.add(tkn)
                    main_anns.append(ent)

        doc.ents = list(doc.ents) + main_anns
