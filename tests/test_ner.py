from medcat.ner.vocab_based_ner import NER
from functools import partial
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.pipe import Pipe
from medcat.utils.normalizers import BasicSpellChecker

config = Config()
config.general['log_level'] = logging.INFO
cdb = CDB(config=config)

# Add a couple of names
cdb.add_names(cui='S-229004', names=prepare_name('Movar', maker.nlp, {}, config))
cdb.add_names(cui='S-229004', names=prepare_name('Movar viruses', maker.nlp, {}, config))
cdb.add_names(cui='S-229005', names=prepare_name('CDB', maker.nlp, {}, config))
# Check
assert cdb.cui2names == {'S-229004': {'movar', 'movarvirus', 'movarviruses'}, 'S-229005': {'cdb'}}

# Make the pipeline
nlp = Pipe(tokenizer=spacy_split_all, config=config)
nlp.add_tagger(tagger=partial(tag_skip_and_punct, config=config),
               name='skip_and_punct',
               additional_fields=['is_punct'])
spell_checker = BasicSpellChecker(cdb_vocab=cdb.vocab, data_vocab=cdb.vocab)
nlp.add_token_normalizer(spell_checker=spell_checker, config=config)
ner = NER(cdb, config)
nlp.add_ner(ner)

# Test limits for tokens and uppercase
config.ner['max_skip_tokens'] = 1
config.ner['upper_case_limit_len'] = 4
text = "CDB - I was running and then Movar    Virus attacked and CDb"
d = nlp(text)

assert len(d._.ents) == 2
assert d._.ents[0]._.link_candidates[0] == 'S-229005'

# Change limit for skip
config.ner['max_skip_tokens'] = 3
d = nlp(text)
assert len(d._.ents) == 3

# Change limit for upper_case
config.ner['upper_case_limit_len'] = 3
d = nlp(text)
assert len(d._.ents) == 4

# Check name length limit
config.ner['min_name_len'] = 4
d = nlp(text)
assert len(d._.ents) == 2

# Speed tests
from timeit import default_timer as timer
text = "CDB - I was running and then Movar    Virus attacked and CDb"
text = text * 300
start = timer()
for i in range(50):
    d = nlp(text)
end = timer()
print("Time: ", end - start)

# Now without spell check
config.general['spell_check'] = False
start = timer()
for i in range(50):
    d = nlp(text)
end = timer()
print("Time: ", end - start)
