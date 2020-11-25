class Linker(object):
    def __init__(self, cdb, config):
        self.cdb = cdb
        self.config = config

    def _train(self, cui, entiy, doc, add_negative=True):
        if force or should_we_skip - re subsampling above 30k?
        self.context_model.train(cui, entity, doc)
            if add_negative and self.config.linking['negative_probability'] >= np.random.rand():
                self.context_model.train_using_negative_sampling(cui)



    def __call__(doc):
        r'''
        '''
        cnf_l = config.linking
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

                    if len(name) >= cnf_l['disamb_len_limit']:
                        if len(cuis) == 1:
                            # N - means name must be disambiguated, is not the prefered
                            #name of the concept, links to other concepts also.
                            if self.cdb.name2cui2status[name][cuis[0]] != 'N':
                                self.context_model.train(cuis[0], entity, doc)
                                entity._.cui = cuis[0]
                                linked_entities.append(entity)
                        else:
                            for cui in cuis:
                                if self.cdb.name2cui2status[name][cui] == 'P':
                                    self.context_model.train(cui, entity, doc)
                                    # It should not be possible that one name is 'P' for two CUIs,
                                    #but it can happen - and we do not care.
                                    entity._.cui = cui
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
                            if len(name) < cnf_l['disamb_len_limit']:
                                do_disambiguate = True
                            elif len(cuis) == 1 and self.cdb.name2cui2status[name][cuis[0]] == 'N':
                                do_disambiguate = True
                            elif len(cuis) > 1:
                                do_disambiguate = True

                            if do_disambiguate:
                                cui, context_similarity = self.context_model.disambiguate(cuis, entity, name, doc)
                            else:
                                cui = cuis[0]
                                if self.config.linking['always_calculate_similarity']:
                                    context_similarity = self.context_model.calculate_similarity(cui, entity, doc)
                                else:
                                    context_similarity = 1 # Direct link, no care for similarity
                    else:
                        # No name detected, just disambiguate
                        cui, context_similarity = self.context_model.disambiguate(entity._.link_candidates, entity, 'unk-unk', doc)

                    # Add the annotation if above threshold
                    if context_similary >= self.config.linking['similarity_threshold']:
                        entity._.cui = cui
                        entity._.context_similarity = context_similarity
                        linked_entities.append(entity)


