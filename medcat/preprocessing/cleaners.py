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


def spacy_tag_punct(doc, skip_stopwords=True, keep_punct=[]):
    for token in doc:
        if IS_PUNCT.match(token.text):
            # There can't be punct in a token
            #if it also has text
            if token.text not in keep_punct:
                token._.is_punct = True
                token._.to_skip = True

        # Skip if specific strings
        if TO_SKIP.match(token.lower_):
            token._.to_skip = True


        # Skip if stopword
        if skip_stopwords and token.is_stop:
            token._.to_skip = True

    return doc
