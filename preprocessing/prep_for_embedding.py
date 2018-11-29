NUM = "NUMNUM"

def clean_and_phrase(iter_data, nlp, out_path):
    """ Will clean and create phrases based on annotations from CAT.
    This expects an iterator over spacy documents.

    iter_data:  iterator over dataset
    out_path:  where to save the new data
    """

    out = open(out_path, 'w')

    for doc in iter_data:
        doc = nlp(doc)

        for ent in doc.ents:
            ent.merge()

        line = ""
        for token in doc:
            if token.is_digit:
                line = line + " " + NUM
            elif not token._.is_punct and len(token.lower_.strip()) > 1:
                if hasattr(token._, 'norm'):
                    tkn = token._.norm
                else:
                    tkn = token.lower_

                line = line + " " + "_".join(tkn.split(" "))

        out.write(line.strip())
        out.write('\n')

    out.close()
