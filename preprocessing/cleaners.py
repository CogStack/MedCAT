""" Text cleaners of various levels, from removing only garbage to
pretty much everything that is not a word.
"""
import re

# Anthing that is not a letter or number is punct,
#for other languages we need to include other letters or normalize input
IS_PUNCT = re.compile(r'[^A-Za-z0-9]+')
TO_SKIP = re.compile(r'^(and|or|nos)$')

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


def clean_umls(text):
    # Remove everything that is inside of []
    text = re.sub("\[[^\]]*\]", "", text)

    # Remove things inside of () if spaces are around it and if the length of the 
    #remaining text is > 15 characters. Stupid approach but works
    tmp = re.sub("(^|\s)\([^\)]*\)($|\s)", " ", text)
    if tmp != text and len(tmp) > 15:
        print(tmp)
        print(text)
        text = tmp

    # Remove multi spaces
    text = re.sub("[ ]+", " ", text).strip()

    return text

def spacy_tag_punct(doc):
    for token in doc:
        if IS_PUNCT.match(token.text):
            # There can't be punct in a token
            #if it also has text
            token._.is_punct = True
        if TO_SKIP.match(token.lower_):
            token._.to_skip = True
    return doc
