import numpy as np
import logging
from typing import Tuple, Dict, List, Union
from spacy.tokens import Span, Doc
from medcat.utils.matutils import unitvec
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config import Config
import random


logger = logging.getLogger(__name__)


class ContextModel(object):
    """Used to learn embeddings for concepts and calculate similarities in new documents.

    Args:
        cdb (CDB): The Context Database
        vocab (Vocab): The vocabulary
        config (Config): The config to be used
    """

    def __init__(self, cdb: CDB, vocab: Vocab, config: Config) -> None:
        self.cdb = cdb
        self.vocab = vocab
        self.config = config

    def get_context_tokens(self, entity: Span, doc: Doc, size: int) -> Tuple:
        """Get context tokens for an entity, this will skip anything that
        is marked as skip in token._.to_skip

        Args:
            entity (Span): The entity to look for.
            doc (Doc): The document look in.
            size (int): The size of the entity.
        """
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

    def get_context_vectors(self, entity: Span, doc: Doc, cui=None) -> Dict:
        """Given an entity and the document it will return the context representation for the
        given entity.

        Args:
            entity (Span): The entity to look for.
            doc (Doc): The document to look in.
            cui (Any): The CUI.

        Returns:
            Dict: The context vector.
        """
        vectors = {}

        for context_type in self.config.linking['context_vector_sizes'].keys():
            size = self.config.linking['context_vector_sizes'][context_type]
            tokens_left, tokens_center, tokens_right = self.get_context_tokens(entity, doc, size)

            values = []
            # Add left
            values.extend([self.config.linking['weighted_average_function'](step) * self.vocab.vec(tkn.lower_)
                           for step, tkn in enumerate(tokens_left) if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])

            if not self.config.linking['context_ignore_center_tokens']:
                # Add center
                if cui is not None and random.random() > self.config.linking['random_replacement_unsupervised'] and self.cdb.cui2names.get(cui, []):
                    new_tokens_center = random.choice(list(self.cdb.cui2names[cui])).split(self.config.general['separator'])
                    values.extend([self.vocab.vec(tkn) for tkn in new_tokens_center if tkn in self.vocab and self.vocab.vec(tkn) is not None])
                else:
                    values.extend([self.vocab.vec(tkn.lower_) for tkn in tokens_center if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])

            # Add right
            values.extend([self.config.linking['weighted_average_function'](step) * self.vocab.vec(tkn.lower_)
                           for step, tkn in enumerate(tokens_right) if tkn.lower_ in self.vocab and self.vocab.vec(tkn.lower_) is not None])

            if len(values) > 0:
                value = np.average(values, axis=0)
                vectors[context_type] = value

        return vectors

    def similarity(self, cui: str, entity: Span, doc: Doc) -> float:
        """Calculate the similarity between the learnt context for this CUI and the context
        in the given `doc`.

        Args:
            cui (str): The CUI.
            entity (Span): The entity to look for.
            doc (Doc): The document to look in.

        Returns:
            float: The simularity.
        """
        vectors = self.get_context_vectors(entity, doc)
        sim = self._similarity(cui, vectors)

        return sim

    def _similarity(self, cui: str, vectors: Dict) -> float:
        """Calculate similarity once we have vectors and a cui.

        Args:
            cui
            vectors
        """

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
                    logger.debug("Similarity for CUI: %s, Count: %s, Context Type: %.10s, Weight: %s.2f, Similarity: %s.3f, S*W: %s.3f",
                                   cui, self.cdb.cui2count_train[cui], context_type, weight, s, s*weight)
            return similarity
        else:
            return -1

    def disambiguate(self, cuis: List, entity: Span, name: str, doc: Doc) -> Tuple:
        vectors = self.get_context_vectors(entity, doc)
        filters = self.config.linking['filters']

        # If it is trainer we want to filter concepts before disambiguation
        #do not want to explain why, but it is needed.
        if self.config.linking['filter_before_disamb']:
            # DEBUG
            logger.debug("Is trainer, subsetting CUIs")
            logger.debug("CUIs before: %s", cuis)

            cuis = [cui for cui in cuis if filters.check_filters(cui)]
            # DEBUG
            logger.debug("CUIs after: %s", cuis)

        if cuis:    # Maybe none are left after filtering
            # Calculate similarity for each cui
            similarities = [self._similarity(cui, vectors) for cui in cuis]
            # DEBUG
            logger.debug("Similarities: %s", [(sim, cui) for sim, cui in zip(cuis, similarities)])

            # Prefer primary
            if self.config.linking.get('prefer_primary_name', 0) > 0:
                logger.debug("Preferring primary names")
                for i, cui in enumerate(cuis):
                    if similarities[i] > 0:
                        if self.cdb.name2cuis2status.get(name, {}).get(cui, '') in {'P', 'PD'}:
                            old_sim = similarities[i]
                            similarities[i] = min(0.99, similarities[i] + similarities[i] * self.config.linking.get('prefer_primary_name', 0))
                            # DEBUG
                            logger.debug("CUI: %s, Name: %s, Old sim: %.3f, New sim: %.3f", cui, name, old_sim, similarities[i])

            if self.config.linking.get('prefer_frequent_concepts', 0) > 0:
                logger.debug("Preferring frequent concepts")
                #Prefer frequent concepts
                cnts = [self.cdb.cui2count_train.get(cui, 0) for cui in cuis]
                m = min(cnts) if min(cnts) > 0 else 1
                scales = [np.log10(cnt/m)*self.config.linking.get('prefer_frequent_concepts', 0) if cnt > 10 else 0 for cnt in cnts]
                similarities = [min(0.99, sim + sim*scales[i]) for i, sim in enumerate(similarities)]

            # Prefer concepts with tag
            mx = np.argmax(similarities)
            return cuis[mx], similarities[mx]
        else:
            return None, 0

    def train(self, cui: str, entity: Span, doc: Doc, negative: bool = False, names: Union[List[str], Dict] = []) -> None:
        """Update the context representation for this CUI, given it's correct location (entity)
        in a document (doc).

        Args:
            names (List[str]/Dict):
                Optionally used to update the `status` of a name-cui pair in the CDB.
        """
        # Context vectors to be calculated
        if len(entity) > 0: # Make sure there is something
            vectors = self.get_context_vectors(entity, doc, cui=cui)
            self.cdb.update_context_vector(cui=cui, vectors=vectors, negative=negative)
            # Debug
            logger.debug("Updating CUI: %s with negative=%s", cui, negative)

            if not negative:
                # Update the name count, if possible
                if type(entity) == Span:
                    self.cdb.name2count_train[entity._.detected_name] = self.cdb.name2count_train.get(entity._.detected_name, 0) + 1

                if self.config.linking.get('calculate_dynamic_threshold', False):
                    # Update average confidence for this CUI
                    sim = self.similarity(cui, entity, doc)
                    self.cdb.update_cui2average_confidence(cui=cui, new_sim=sim)

            if negative:
                # Change the status of the name so that it has to be disambiguated always
                for name in names:
                    if self.cdb.name2cuis2status.get(name, {}).get(cui, '') == 'P':
                        # Set this name to always be disambiguated, even though it is primary
                        self.cdb.name2cuis2status.get(name, {})[cui] = 'PD'
                        # Debug
                        logger.debug("Updating status for CUI: %s, name: %s to <PD>", cui, name)
                    elif self.cdb.name2cuis2status.get(name, {}).get(cui, '') == 'A':
                        # Set this name to always be disambiguated instead of A
                        self.cdb.name2cuis2status.get(name, {})[cui] = 'N'
                        logger.debug("Updating status for CUI: %s, name: %s to <N>", cui, name)
            if not negative and self.config.linking.get('devalue_linked_concepts', False):
                #Find what other concepts can be disambiguated against this one
                _cuis = set()
                for name in self.cdb.cui2names[cui]:
                    _cuis.update(self.cdb.name2cuis.get(name, []))
                # Remove the cui of the current concept
                _cuis = _cuis - {cui}

                for _cui in _cuis:
                    self.cdb.update_context_vector(cui=_cui, vectors=vectors, negative=True)

                logger.debug("Devalued via names.\n\tBase cui: %s \n\tTo be devalued: %s\n", cui, _cuis)
        else:
            logger.warning("The provided entity for cui <%s> was empty, nothing to train", cui)

    def train_using_negative_sampling(self, cui: str) -> None:
        vectors = {}

        # Get vectors for each context type
        for context_type in self.config.linking['context_vector_sizes'].keys():
            size = self.config.linking['context_vector_sizes'][context_type]
            # While it should be size*2 it is already too many negative examples, so we leave it at size
            inds = self.vocab.get_negative_samples(size, ignore_punct_and_num=self.config.linking['negative_ignore_punct_and_num'])
            values = [self.vocab.vec(self.vocab.index2word[ind]) for ind in inds]
            if len(values) > 0:
                vectors[context_type] = np.average(values, axis=0)
            # Debug
            logger.debug("Updating CUI: %s, with %s negative words", cui, len(inds))

        # Do the update for all context types
        self.cdb.update_context_vector(cui=cui, vectors=vectors, negative=True)
