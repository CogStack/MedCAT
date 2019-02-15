import pandas
import re

NUM = "NUMNUM"

FAST_SPLIT = re.compile("[^A-Za-z0-9]")


class EmbMimicCSV(object):
    """ Iterate over MIMIC data in CSV format

    csv_paths:  paths to csv files containing the mimic data
    """
    def __init__(self, csv_paths, tokenizer, emb_dict=None):
        self.csv_paths = csv_paths
        self.tokenizer = tokenizer
        self.emb_dict = emb_dict

    def __iter__(self):
        chunksize = 10 ** 8
        for csv_path in self.csv_paths:
            for chunk in pandas.read_csv(csv_path, chunksize=chunksize):
                for _, row in chunk.iterrows():
                    doc = self.tokenizer(row['text'])
                    data = []
                    for token in doc:
                        if not token._.is_punct and not token._.to_skip and len(token.lower_.strip()) > 1:
                            if token.is_digit:
                                data.append(NUM)
                            else:
                                if hasattr(token._, 'norm'):
                                    tkn = token._.norm
                                else:
                                    tkn = token.lower_

                                if self.emb_dict is not None:
                                    if tkn in self.emb_dict:
                                        data.append(tkn)
                                else:
                                    data.append(tkn)
                    yield data


class BertEmbMimicCSV(object):
    """ Iterate over MIMIC data in CSV format

    csv_paths:  paths to csv files containing the mimic data
    """
    def __init__(self, csv_paths, tokenizer):
        from pytorch_pretrained_bert import BertTokenizer

        self.csv_paths = csv_paths
        self.tokenizer = tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __iter__(self):
        chunksize = 10 ** 8
        for csv_path in self.csv_paths:
            for chunk in pandas.read_csv(csv_path, chunksize=chunksize):
                for _, row in chunk.iterrows():
                    doc = self.tokenizer(row['text'])
                    data = []
                    for token in doc:
                        tkn = token._.lower

                        for tkn in self.bert_tokenizer.tokenize(tkn):
                            data.append(tkn)
                    yield data



class RawCSV(object):
    """ Iterate over MIMIC data in CSV format

    csv_paths:  paths to csv files containing the mimic data
    """
    def __init__(self, csv_paths):
        self.csv_paths = csv_paths

    def __iter__(self):
        chunksize = 10 ** 8
        for csv_path in self.csv_paths:
            for chunk in pandas.read_csv(csv_path, chunksize=chunksize):
                for _, row in chunk.iterrows():
                    yield row['text']


class FastEmbMimicCSV(object):
    """ Iterate over MIMIC data in CSV format

    csv_paths:  paths to csv files containing the mimic data
    """
    def __init__(self, csv_paths):
        self.csv_paths = csv_paths

    def __iter__(self):
        chunksize = 10 ** 8
        for csv_path in self.csv_paths:
            for chunk in pandas.read_csv(csv_path, chunksize=chunksize):
                for _, row in chunk.iterrows():
                    doc = [x for x in FAST_SPLIT.split(row['text']) if len(x) > 0]
                    doc = [x.lower() if not x.isdigit() else NUM for x in doc]
                    yield doc



