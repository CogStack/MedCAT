""" Utils for building vocabs from text files
"""

import spacy


def text2vocab(in_path, out_path, cleaner=None, length_limit=0):
    """ Build a vocab from a text file

    in_path:  Input txt file, each line is one sentence.
    out_path:  Where to save the built vocab, each line is one word
    cleaner:  Function used for cleaning the text file, if None no cleaning
               will be done
    length_limit:  Words with length < than this will not be added to the vocab
    """

    in_f = open(in_path, 'r')
    nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
    vocab = set()

    cnt = 0
    for line in in_f:
        if cleaner is not None:
            line = cleaner(line)

        doc = nlp(line)
        for token in doc:
            if len(token.norm_) >= length_limit:
                vocab.add(token.norm_)

        cnt += 1
        if cnt % 100000 == 0:
            print("Processed: " + str(cnt))

    with open(out_path, 'w') as out_f:
        for word in sorted(vocab):
            out_f.write(word + "\n")
