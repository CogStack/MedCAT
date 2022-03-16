import random
import logging
import math

from enum import Enum, auto
from spacy.tokens import Span, Doc
from typing import Optional, List, Dict
from medcat.utils.filters import check_filters
from medcat.linking.vector_context_model import ContextModel
from medcat.pipeline.pipe_runner import PipeRunner
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config import Config


class LabelStyle(Enum):
    short = auto()
    long = auto()


class Linker(PipeRunner):
    r''' Link to a biomedical database.

    Args:
        cdb
        vocab
        config
    '''
    log = logging.getLogger(__name__)

    # Custom pipeline component name
    name = 'cat_linker'

    # Override
    def __init__(self, cdb: CDB, vocab: Vocab, config: Config) -> None:
        self.cdb = cdb
        self.vocab = vocab
        self.config = config
        self.context_model = ContextModel(self.cdb, self.vocab, self.config)
        # Counter for how often did a pair (name,cui) appear and was used during training
        self.train_counter: Dict = {}
        super().__init__(self.config.general['workers'])

    def _train(self, cui: str, entity: Span, doc: Doc, add_negative: bool = True) -> None:
        name = "{} - {}".format(entity._.detected_name, cui)
        if self.train_counter.get(name, 0) > self.config.linking['subsample_after']:
            if random.random() < 1 / math.sqrt(self.train_counter.get(name) - self.config.linking['subsample_after']):
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

    # Override
    def __call__(self, doc: Doc) -> Doc:
        r'''
        '''
        cnf_l = self.config.linking
        linked_entities = []

        if cnf_l["train"]:
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
                self.log.debug("Linker started with entity: %s", entity)
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
                    if cui and check_filters(cui, self.config.linking['filters']):
                        th_type = self.config.linking.get('similarity_threshold_type', 'static')
                        if (th_type == 'static' and context_similarity >= self.config.linking['similarity_threshold']) or \
                           (th_type == 'dynamic' and context_similarity >= self.cdb.cui2average_confidence[cui] * self.config.linking['similarity_threshold']):
                            entity._.cui = cui
                            entity._.context_similarity = context_similarity
                            linked_entities.append(entity)

        doc._.ents = linked_entities
        self._create_main_ann(doc)

        if self.config.general['make_pretty_labels'] is not None:
            self._make_pretty_labels(doc, LabelStyle[self.config.general['make_pretty_labels']])

        if self.config.general['map_cui_to_group'] is not None and self.cdb.addl_info.get('cui2group', {}):
            self._map_ents_to_groups(doc)

        return doc

    def _map_ents_to_groups(self, doc: Doc) -> None:
        for ent in doc.ents:
            ent._.cui = self.cdb.addl_info['cui2group'].get(ent._.cui, ent._.cui)

    def _make_pretty_labels(self, doc: Doc, style: Optional[LabelStyle] = None) -> None:
        ents = list(doc.ents)

        n_ents = []
        for ent in ents:
            if style == LabelStyle.short:
                label = ent._.cui
            elif style == LabelStyle.long:
                label = "{} | {} | {:.2f}".format(ent._.cui, self.cdb.get_name(ent._.cui), ent._.context_similarity)
            else:
                label = 'concept'

            n_ent = Span(doc, ent.start, ent.end, label)
            for attr in ent._.__dict__['_extensions'].keys():
                setattr(n_ent._, attr, getattr(ent._, attr))
            n_ents.append(n_ent)

        doc.ents = n_ents

    def _create_main_ann(self, doc: Doc, tuis: Optional[List] = None) -> None:
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
