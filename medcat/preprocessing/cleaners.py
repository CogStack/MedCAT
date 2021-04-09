""" Text cleaners of various levels, from removing only garbage to
pretty much everything that is not a word.
"""
import re

def prepare_name(raw_name, nlp, names, config):
    r''' Generates different forms of a name. Will edit the provided `names` dictionary
    and add information generated from the `name`.

    Args:
        nlp (`spacy.lang.<lng>`):
            Spacy nlp model.
        names (`dict`):
            Dictionary of existing names for this concept in this row of a CSV. The new generated
            name versions and other required information will be added here.
        config (`medcat.config.Config`):
            Global config for medcat.

    Return:
        names (`dict`):
            The new dictionary of prepared names.
    '''
    sc_name = nlp(raw_name)

    for version in config.cdb_maker['name_versions']:
        tokens = None

        if version == "LOWER":
            tokens = [t.lower_ for t in sc_name if not t._.to_skip]
        if version == "CLEAN":
            tokens = []
            for t in sc_name:
                if not t._.to_skip:
                    if len(t.lower_) < config.preprocessing['min_len_normalize']:
                        tokens.append(t.lower_)
                    elif (config.preprocessing.get('do_not_normalize', set())) and t.tag_ is not None and \
                            t.tag_ in config.preprocessing.get('do_not_normalize'):
                        tokens.append(t.lower_)
                    else:
                        tokens.append(t.lemma_.lower())

        if tokens is not None and tokens:
            snames = set()
            name = config.general['separator'].join(tokens)

            if not config.cdb_maker.get('min_letters_required', 0) or len(re.sub("[^A-Za-z]*", '', name)) >= config.cdb_maker.get('min_letters_required'):
                if name not in names:
                    sname = ""
                    for token in tokens:
                        if sname:
                            sname = sname + config.general['separator'] + token
                        else:
                            sname = token
                        snames.add(sname.strip())

                    names[name] = {'tokens': tokens, 'snames': snames, 'raw_name': raw_name}

    return names


def basic_clean(text):
    """ Remove almost everything from text

    text:  text to be cleaned
    """
    # Add spaces around numbers
    text = re.sub("([\.,%:\d\-]*[\d]+[\.,%:\d\-]*)", r' \1 ', text)

    # Remove some chars
    text = re.sub("[:;\\|!?%#@%\&=><\-\*\+\^]", " ", text)

    # Remove dots not preeceded by a letter or number
    text = re.sub("[^A-Za-z0-9]+\.", "", text)

    # Remove commas not in-between numbers
    text = re.sub(",([^0-9])|([^0-9]),", r"\2\1 ", text)

    # Remove multi-spaces and tabs
    text = re.sub("\t+", " ", text)
    text = re.sub("[ ]+", " ", text)

    # Remove any character that appears more than 2 times in a row,
    #unless it is a number
    text = re.sub(r'([^0-9]{1})\1{2,}', r'\1\1', text)

    return text.strip().lower()


def clean_text(text):
    """ Remove almost everything from text

    text:  text to be cleaned
    """
    # Remove everything that is inside of []
    text = re.sub("\[.*\]", "", text)

    # Information in () not important, this will leave () if we have line breaks
    text = re.sub("\(.*\)", "", text)

    # Remove numbers, nlp is not really good with numbers in this case
    #text = re.sub("[\.,%:\d\-]*[\d]+[\.,%:\d\-]*", " NUM ", text)

    # Add spaces around numbers
    text = re.sub("([\.,%:\d\-]*[\d]+[\.,%:\d\-]*)", r' \1 ', text)

    # Remove some chars
    text = re.sub("\/", " ", text)
    text = re.sub("[:;\\|!?%#@%\&=><\-\*\+\^]", " ", text)

    # Remove dots not preeceded by a letter or number
    text = re.sub("[^A-Za-z0-9]+\.", "", text)

    # Remove commas not in-between numbers
    text = re.sub(",([^0-9])|([^0-9]),", r"\2\1 ", text)

    # Remove multi-spaces and tabs
    text = re.sub("\t+", " ", text)
    text = re.sub("[ ]+", " ", text)

    # Remove any character that appears more than 2 times in a row
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    return text.strip().lower()


BR_U4 = re.compile("\[[^\]]{0,3}\]")
CB = re.compile("(\s)\([a-zA-Z]+[^\)\(]*\)(\s)")
CB_D = re.compile("(\s)\([a-z]+[^\)\(]*\)($)")
BR = re.compile("(^|\s)\[[^\]]*\]($|\s)")
PH_RM = re.compile("(\(|\[)(observation|finding|symptoms|disease|observations|disorder|disease/finding)(\)|\])", flags=re.I)
SKIP_CHARS = re.compile("[\[\]\*]+")


def clean_drugs_uk(text, stopwords=None, umls=None):
    _text = CB.sub(" ", text)
    _text = CB.sub(" ", _text)
    _text = CB_D.sub(" ", _text)
    if len(_text) > 8:
        text = _text

    return clean_name(text, stopwords, umls)


def clean_name(text, stopwords=None, umls=False):
    # Remove multi spaces
    text = re.sub("[ ]+", " ", text).strip()

    # If UMLS
    if umls:
        # Remove specific things from parentheses
        text = PH_RM.sub(" ", text)

    # Remove stopwords if requested and <= 5 words in total in the name
    if stopwords:
        new_text = ""
        for word in text.split(" "):
            if word not in stopwords:
                new_text += word + " "
        text = new_text.strip()


    return text



def clean_umls(text, stopwords=None):
    # Remove [] if < 4 letters inside
    text = BR_U4.sub(" ", text)

    # Remove things inside of () or [] if spaces are around it and if the length of the 
    #remaining text is > 15 characters. Stupid approach but works
    #tmp = CB.sub(" ", text)
    #tmp = BR.sub(" ", tmp)
    #if tmp != text and len(tmp) > 15:
    #    text = tmp

    # Remove specific things from parentheses
    text = PH_RM.sub(" ", text)

    # Remove multi spaces
    text = re.sub("[ ]+", " ", text).strip()

    # Remove stopwords if requested and <= 5 words in total in the name
    if stopwords:
        new_text = ""
        for word in text.split(" "):
            if word not in stopwords:
                new_text += word + " "
        text = new_text.strip()

    return text

def clean_def(text):
    # Remove things inside of () or [] 
    text = re.sub("\([^\)]*\)", " ", text)
    text = re.sub("\[[^\]]*\]", " ", text)

    # Remove multi spaces
    text = re.sub("[ ]+", " ", text).strip()

    return text

def clean_snt(text):
    # Remove things inside of () or [] 
    text = re.sub("\[\*[^\]]*\*\]", " ", text)

    # Remove multi _-
    text = re.sub("[_-]{2,}", " ", text)

    # Remove multi spaces
    text = re.sub("[ ]+", " ", text).strip()

    return text

def clean_snomed_name(text):
    # Remove () from end of string
    text = text.strip()
    text = re.sub("\([^\)]*\)$", " ", text).strip()

    return text
