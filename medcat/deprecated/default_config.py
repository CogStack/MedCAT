class Config(object):
    def __init__(self, **kwargs):
        # Is the pipeline in the debug mode or not, debug mainly means it will
        #log a lot of stuff
        self.debug = kwargs.pop("debug", False)
        # What punctuation to NOT ignore during annotation, by default all punct is ignored
        self.keep_punct = kwargs.pop("keep_punct", ['.', ':'])
        # Do we want to have nested entities in the JSON output
        self.nested_entities = kwargs.pop("nested_entities", False)
        # When calculating the context embedding to we want normalized tokens (tkn._.norm) or just lowercased (tkn.text.lower())
        self.norm_emb = kwargs.pop("norm_emb", False)
        # If True, concepts that appear frequently will be prefered during disambiguation
        self.prefer_frequent = kwargs.pop("prefer_frequent", False)
        # If True, concepts that have an ICD10 code will be prefer during disambiguation
        self.prefer_icd10 = kwargs.pop("prefer_icd10", False)
        # Window size when calculating the context embedding (N - from left, N from right)
        self.cntx_span = kwargs.pop("cntx_span", 9)
        # Same as CNTX_SPAN, but usually 1/3 of the size
        self.cntx_span_short = kwargs.pop("cntx_span_short", 3)
        # If a concept appeared below this number it will always be trained in the unsuperwised mode, above this number
        #the concept will be trained only sometimes with decreasing train probability depending on the number of occurences.
        self.min_cui_count = kwargs.pop("min_cui_count", 30000)
        # Minimum number of trainign examples a concept must have before it can be considered 
        #for disambiguation
        self.disambiguation_limit = kwargs.pop("disambiguation_limit", 1)
        # By default medcat will not calculate the accuracy for concepts that are not ambiguous (acc=1 for them), but you can enable
        #that by setting the below to True.
        self.acc_always = kwargs.pop("acc_always", False)
        # Very similar to acc_always, but this setting will disambiguate all concepts no matter are there really
        #ambiguous or not
        self.disamb_everything = kwargs.pop("disamb_everything", False)
        # When detecting entity what is the maximum number of tokens that can be skipped/ignored while still considering 
        #the span to be one entity. Used for multi-word entites that possible can have additional words in the text that we can ignore. 
        self.max_skip_tkn = kwargs.pop("max_skip_tkn", 2)
        # Do we ignore stopwords or not
        self.skip_stopwords = kwargs.pop("skip_stopwords", False)
        # When calculating the context embedding do we want to use weighted average or simply average
        self.weighted_average = kwargs.pop("weighted_average", True)
        # Minimu accuracy for concept detection, concepts below will not be shown/returned
        self.min_acc = kwargs.pop("min_acc", 0.2)
        # Mimum length of the concept span that can be detected
        self.min_concept_length = kwargs.pop("min_concept_length", 1)
        # Probability of adding a negative context each time we add a positive one
        self.neg_prob = kwargs.pop("neg_prob", 0.5)
        # Style of the labels, currently supported long/ent/none
        self.lbl_style = kwargs.pop("lbl_style", 'ent').lower()
