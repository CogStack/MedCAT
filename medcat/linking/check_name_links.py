import numpy as np
from medcat.utils.matutils import unitvec
from medcat.utils.loggers import basic_logger

class AnnotationChecker(object):
    def __init__(self, cdb, vocab, config):
        self.cdb = cdb
        self.vocab = vocab
        self.config = config

        self.log = basic_logger(name='cdb', config=self.config)


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

        tokens_left = [tkn for tkn in doc[max(0, start_ind-size):start_ind] if not tkn._.to_skip]
        # Reverse because the first token should be the one closest to center
        tokens_left.reverse()
        tokens_center = list(entity)
        tokens_right = [tkn for tkn in doc[end_ind+1:end_ind + 1 + size] if not tkn._.to_skip]

        return tokens_left, tokens_center, tokens_right


    def get_context_vectors(self, entity, doc):
        vectors = {}

        for context_type in self.config.linking['context_vector_sizes'].keys():
            size = self.config.linking['context_vector_sizes'][context_type]
            tokens_left, tokens_center, tokens_right = self.get_context_tokens(entity, doc, size)

            values = []
            # Add left
            values.extend([self.config.linking['weighted_average_function'](step) * self.vocab.vec(tkn.lower_)
                           for step, tkn in enumerate(tokens_left) if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])
            # Add center
            values.extend([self.vocab.vec(tkn.lower_) for tkn in tokens_left if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])

            # Add right
            values.extend([self.config.linking['weighted_average_function'](step) * self.vocab.vec(tkn.lower_)
                           for step, tkn in enumerate(tokens_left) if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])

            if len(values) > 0:
                value = np.average(values, axis=0)
                vectors[context_type] = value

        return vectors

    def calculate_similarity(self, cui, entity, doc):
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

        cui_vectors = self.cdb.cui2context_vectors.get(cui, None)

        if cui_vectors is not None and self.cdb.cui2count_train[cui] >= self.config.linking['train_count_threshold']:
            similarity = 0
            for context_type in self.config.linking['context_vector_weights']:
                weight = self.config.linking['context_vector_weights'][context_type]
                s = np.dot(unitvec(vectors[context_type]), unitvec(cui_vectors[context_type]))
                similarity += weight * s

                # DEBUG
                self.log.debug("Similarity for CUI: {}, Count: {}, Context Type: {:10}, Weight: {:.2f}, Similarity: {:.3f}, S*W: {:.3f}".format(
                    cui, self.cdb.cui2count_train[cui], context_type, weight, s, s*weight))
            return similarity
        else:
            return -1

    def disambiguate_and_filter(self, cuis, entity, name, doc):
        vectors = self.get_context_vectors(entity, doc)

        # ADD FILTERS
        if filters:
            if self.config.general['is_trainer']:
                cuis = None
            else:
                cuis = # Filter
        for cui in cuis:
            sim = self._similarity(cui, vectors)

            if 
        # Prefer frequent

        # Prefer primary

        # Prefer concepts with tag


    def train(self, cui, entity, doc, negative=False):
        # Context vectors to be calculated
        vectors = self.get_context_vectors(entity, doc)
        self.cdb.update_context_vector(cui=cui, vectors=vectors, negative=negative)


    def train_using_negative_sampling(self, cui):
        vectors = {}
        for context_type in self.config.linking['context_vector_sizes'].keys():
            size = self.config.linking['context_vector_sizes'][context_type]
            # While it should be size*2 it is already too many negative examples, so we leave it at size
            inds = self.vocab.get_negative_samples(size, ignore_punct_and_num=self.config.linking['negative_ignore_punct_and_num'])
            values = [self.vocab.vec(self.vocab.index2word[ind]) for ind in inds]
            vectors[context_type] = np.average(values, axis=0)

        self.cdb.update_context_vector(cui=cui, vectors=vectors, negative=True)
