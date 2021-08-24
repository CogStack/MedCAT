from ssl import Purpose
from medcat.pipe import Pipe
from medcat.vocab import Vocab

from typing import Dict, List, Set
import spacy

class RelationalModel(object):

   #def create_instances(max_length: int) -> Callable[[Doc], List[Tuple[Span, Span]]]:
   #    def get_instances(doc: Doc) -> List[Tuple[Span, Span]]:
   #        instances = []
   #        for ent1 in doc.ents:
   #            for ent2 in doc.ents:
   #                if ent1 != ent2:
   #                    if max_length and abs(ent2.start - ent1.start) <= max_length:
   #                        instances.append((ent1, ent2))
   #        return instances
   #    return get_instances

    def __init__(self):
        pass

    def get_instances(self, doc, max_length=10000):
       instances = []
       for ent1 in doc.ents:
           for ent2 in doc.ents:
               if ent1 != ent2:
                   if max_length and abs(ent2.start - ent1.start) <= max_length:
                       instances.append((ent1, ent2))
       return instances

class RelationExtraction(Pipe):
   def __init__(self, vocab: Vocab,  model: RelationalModel = None, threshold: float = 0.1, name: str = "rel"):
       self.vocab = vocab
       self.model = model
       self.name = name
       self.cfg = {"labels": [], "threshold": threshold }