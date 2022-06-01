def deid_text(cat, text, redact=False):
    new_text = str(text)
    entities = cat.get_entities(text)['entities']
    for ent in sorted(entities.values(), key=lambda ent: ent['start'], reverse=True):
        r = "*"*(ent['end']-ent['start']) if redact else cat.cdb.get_name(ent['cui'])
        new_text = new_text[:ent['start']] + f'[{r}]' + new_text[ent['end']:]
    return new_text
