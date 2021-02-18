import pandas as pd
import regex


def normalize_date(date):
    """ Normalizes different dates encountered in the clinical notes.
    Current accepted formats:
        Thu 28 Feb 2013 04:50
        28-Feb-2013 04:50
    Output:
        28 Feb 2013 04:50
    """

    if '-' in date:
        date = date.replace("-", " ").strip()
    elif date.strip()[0].isalpha():
        date = date[date.index(' '):].strip()
    else:
        print("Unsuported date format: {}".format(date))

    return date


def split_one_note(id, text):
    """ Splits the text of one note by date.

    Return:
        split_note (List[Dict]):
            Returns a list of dictionary in the format: {'start': <start char of the specific note in the big one>,
                                                         'end': <end char of the specifc note in the big one>,
                                                         'text': <text of the specific note>,
                                                         'date': <date of the specific note>}
    """
    r = r'\n\w{0,5}\s*\d{1,2}(\s|-)[a-zA-Z]{3,5}(\s|-)\d{4}\s+\d{2}\:\d{2}'
    dates = regex.finditer(r, text)
    start = 0
    end = -1
    split_note = []
    previous_date = None

    for date in dates:
        if start == 0:
            start = date.span()[0]
            previous_date = date.captures()[0]
        elif previous_date is None or date.captures()[0] != previous_date:
            end = date.span()[0]
            note_text = text[start:end]
            if 'entered on -' in note_text.lower():
                if len(regex.findall(r'entered on -', note_text)) > 1:
                    print("Possible problems for span with start: {} and end: {} for note with id: {}".format(start, end, id))
                split_note.append({'start': start, 'end': end, 'text': note_text, 'date': normalize_date(previous_date)})
                start = end
                previous_date = date.captures()[0]
    # Add the last note
    if previous_date is not None:
        split_note.append({'start': start, 'end': len(text), 'text': text[start:], 'date': normalize_date(previous_date)})
    else:
        print(text)

    return split_note


def split_clinical_notes(clinical_notes):
    """ Splits clinical notes.

    Args:
        clinical_notes(dict):
            Dictionary in the form {<clinical_note_id>: <text>, ...}
    """
    split_notes = {}
    for id, text in clinical_notes.items():
        split_notes[id] = split_one_note(id, text)
    return split_notes
