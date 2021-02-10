import numpy as np
import logging
from medcat.utils.matutils import unitvec
from medcat.utils.filters import check_filters

class ContextModel(object):
    r''' Used to learn embeddings for concepts and calculate similarities in new documents.

    Args:
        cdb
        vocab
        config
    '''
    log = logging.getLogger(__name__)
    def __init__(self, cdb, vocab, config):
        self.cdb = cdb
        self.vocab = vocab
        self.config = config

    def get_context_tokens(self, entity, doc, size):
        r''' Get context tokens for an entity, this will skip anything that
        is marked as skip in token._.to_skip

        Args:
            entity
            doc
            size
        '''
        start_ind = entity[0].i
        end_ind = entity[-1].i

        tokens_left = [tkn for tkn in doc[max(0, start_ind-size):start_ind] if not tkn._.to_skip
                and not tkn.is_stop and not tkn.is_digit and not tkn.is_punct]
        # Reverse because the first token should be the one closest to center
        tokens_left.reverse()
        tokens_center = list(entity)
        tokens_right = [tkn for tkn in doc[end_ind+1:end_ind + 1 + size] if not tkn._.to_skip
                and not tkn.is_stop and not tkn.is_digit and not tkn.is_punct]

        return tokens_left, tokens_center, tokens_right


    def get_context_vectors(self, entity, doc):
        r''' Given an entity and the document it will return the context representation for the
        given entity.

        Args:
            entity
            doc
        '''
        vectors = {}

        for context_type in self.config.linking['context_vector_sizes'].keys():
            size = self.config.linking['context_vector_sizes'][context_type]
            tokens_left, tokens_center, tokens_right = self.get_context_tokens(entity, doc, size)

            values = []
            # Add left
            values.extend([self.config.linking['weighted_average_function'](step) * self.vocab.vec(tkn.lower_)
                           for step, tkn in enumerate(tokens_left) if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])
            # Add center - used normalized
            values.extend([self.vocab.vec(tkn._.norm) for tkn in tokens_center if tkn._.norm in self.vocab and self.vocab.vec(tkn._.norm) is not None])

            # Add right
            values.extend([self.config.linking['weighted_average_function'](step) * self.vocab.vec(tkn.lower_)
                           for step, tkn in enumerate(tokens_right) if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])

            if len(values) > 0:
                value = np.average(values, axis=0)
                vectors[context_type] = value

        return vectors

    def similarity(self, cui, entity, doc):
        r''' Calculate the similarity between the learnt context for this CUI and the context
        in the given `doc`.

        Args:
            cui
            entity
            doc
        '''
        vectors = self.get_context_vectors(entity, doc)
        sim = self._similarity(cui, vectors)

        return sim


    def _similarity(self, cui, vectors):
        r''' Calculate similarity once we have vectors and a cui.

        Args:
            cui
            vectors
        '''

        cui_vectors = self.cdb.cui2context_vectors.get(cui, {})

        if cui_vectors and self.cdb.cui2count_train[cui] >= self.config.linking['train_count_threshold']:
            similarity = 0
            for context_type in self.config.linking['context_vector_weights']:
                # Can be that a certain context_type does not exist for a cui/context
                if context_type in vectors and context_type in cui_vectors:
                    weight = self.config.linking['context_vector_weights'][context_type]
                    s = np.dot(unitvec(vectors[context_type]), unitvec(cui_vectors[context_type]))
                    similarity += weight * s

                    # DEBUG
                    self.log.debug("Similarity for CUI: {}, Count: {}, Context Type: {:10}, Weight: {:.2f}, Similarity: {:.3f}, S*W: {:.3f}".format(
                        cui, self.cdb.cui2count_train[cui], context_type, weight, s, s*weight))
            return similarity
        else:
            return -1

    def disambiguate(self, cuis, entity, name, doc):
        vectors = self.get_context_vectors(entity, doc)
        filters = self.config.linking['filters']

        # If it is trainer we want to filter concepts before disambiguation
        #do not want to explain why, but it is needed.
        if self.config.linking['filter_before_disamb']:
            # DEBUG
            self.log.debug("Is trainer, subsetting CUIs")
            self.log.debug("CUIs before: {}".format(cuis))

            cuis = [cui for cui in cuis if check_filters(cui, filters)]
            # DEBUG
            self.log.debug("CUIs after: {}".format(cuis))

        if cuis: #Maybe none are left after filtering
            # Calculate similarity for each cui
            similarities = [self._similarity(cui, vectors) for cui in cuis]
            # DEBUG
            self.log.debug("Similarities: {}".format([(sim, cui) for sim,cui in zip(cuis, similarities)]))

            # Prefer primary
            if self.config.linking['prefer_primary_name']:
                self.log.debug("Preferring primary names")
                for i, cui in enumerate(cuis):
                    if similarities[i] > 0:
                        if self.cdb.name2cuis2status.get(name, {}).get(cui, '') in {'P', 'PD'}:
                            old_sim = similarities[i]
                            similarities[i] = min(0.99, similarities[i] + similarities[i] * 0.3)
                            # DEBUG
                            self.log.debug("CUI: {}, Name: {}, Old sim: {:.3f}, New sim: {:.3f}".format(cui, name, old_sim, similarities[i]))

            # Prefer concepts with tag
            mx = np.argmax(similarities)
            return cuis[mx], similarities[mx]
        else:
            return None, 0


    def train(self, cui, entity, doc, negative=False, names=[]):
        r''' Update the context represenation for this CUI, given it's correct location (entity)
        in a document (doc).

        Args:
            names (List[str]/Dict):
                Optinaly used to update the `status` of a name-cui pair in the CDB.
        '''
        # Context vectors to be calculated
        if len(entity) > 0: # Make sure there is something
            vectors = self.get_context_vectors(entity, doc)
            self.cdb.update_context_vector(cui=cui, vectors=vectors, negative=negative)
            # Debug
            self.log.debug("Updating CUI: {} with negative={}".format(cui, negative))

            if negative:
                # Change the status of the name so that it has to be disambiguated always
                for name in names:
                    if self.cdb.name2cuis2status.get(name, {}).get(cui, '') == 'P':
                        # Set this name to always be disambiguated, even though it is primary
                        self.cdb.name2cuis2status.get(name, {})[cui] = 'PD'
                        # Debug
                        self.log.debug("Updating status for CUI: {}, name: {} to <PD>".format(cui, name))
                    elif self.cdb.name2cuis2status.get(name, {}).get(cui, '') == 'A':
                        # Set this name to always be disambiguated instead of A
                        self.cdb.name2cuis2status.get(name, {})[cui] = 'N'
                        self.log.debug("Updating status for CUI: {}, name: {} to <N>".format(cui, name))
            if not negative and self.config.linking.get('devalue_linked_concepts', False):
                #Find what other concepts can be disambiguated against this one
                _cuis = set()
                for name in self.cdb.cui2names[cui]:
                    _cuis.update(self.cdb.name2cuis.get(name, []))
                # Remove the cui of the current concept
                _cuis = _cuis - {cui}

                for _cui in _cuis:
                    self.cdb.update_context_vector(cui=_cui, vectors=vectors, negative=True)

                self.log.debug("Devalued via names.\n\tBase cui: {} \n\tTo be devalued: {}\n".format(cui, _cuis))
        else:
            self.log.warning("The provided entity for cui <{}> was empty, nothing to train".format(cui))


    def train_using_negative_sampling(self, cui):
        vectors = {}
        for context_type in self.config.linking['context_vector_sizes'].keys():
            size = self.config.linking['context_vector_sizes'][context_type]
            # While it should be size*2 it is already too many negative examples, so we leave it at size
            inds = self.vocab.get_negative_samples(size, ignore_punct_and_num=self.config.linking['negative_ignore_punct_and_num'])
            values = [self.vocab.vec(self.vocab.index2word[ind]) for ind in inds]
            if len(values) > 0:
                vectors[context_type] = np.average(values, axis=0)
            # Debug
            self.log.debug("Updating CUI: {}, with {} negative words".format(cui, len(inds)))

        self.cdb.update_context_vector(cui=cui, vectors=vectors, negative=True)
