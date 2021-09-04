import jsonpickle


class ConfigMetaCAT(object):
    jsonpickle.set_encoder_options('json', sort_keys=True, indent=2)

    def __init__(self):
        self.general = {
                'device': 'cuda',
                'seed': 13,
                'category_name': None, # What category is this meta_cat model predicting/training
                'category_value2id': {}, # Map from category values to ID, if empty it will be autocalculated during training
                'lowercase': True, # If true all input text will be lowercased
                'cntx_in_chars': False, # If True the context size is measured in number of characters, otherwise in tokens
                'cntx_left': 15,
                'cntx_right': 10,
                'replace_center': None, # If set the center (concept) will be replaced with this string
                }
        self.model = {
                'model_name': 'lstm',
                'num_layers': 2,
                'input_size': 300,
                'hidden_size': 300,
                'dropout': 0.5,
                'num_directions': 2, # 2 - bidirectional model, 1 - unidirectional
                'nclasses': 2, # Number of classes that this model will output
                'padding_idx': 30000,
                'emb_grad': True, # If True the embeddings will also be trained
                'ignore_cpos': False, # If set to True center positions will be ignored when calculating represenation
                }

        self.train = {
                'batch_size': 100,
                'nepochs': 20,
                'lr': 0.01,
                'test_size': 0.1,
                'class_weights': None,
                'score_average': 'weighted', # What to use for averaging F1/P/R across labels
                'prerequisites': {},
                'cui_filter': None, # If set only this CUIs will be used for training
                'auto_save_model': True, # Should do model be saved during training for best results
                }


    def save(self, save_path):
        r''' Save the config into a .json file

        Args:
            save_path (`str`):
                Where to save the created json file
        '''
        # We want to save the dict here, not the whole class
        json_string = jsonpickle.encode(self.__dict__)

        with open(save_path, 'w') as f:
            f.write(json_string)


    def merge_config(self, config_dict):
        r''' We update the current class dictionary with the loaded one,
        very strange solution - idea is to keep new fields that were potentially added
        after this config was created, but to overwrite fields that exist in the loaded
        config.

        Args:
            config_dict (`dict`):
                A dictionary which key/values should be added to this class.
        '''
        for key in config_dict.keys():
            if key in self.__dict__:
                self.__dict__[key].update(config_dict[key])
            else:
                self.__dict__[key] = config_dict[key]


    @classmethod
    def load(cls, save_path):
        r''' Load config from a json file, note that fields that
        did not exist in the old config but do exist in the current
        version of the ConfigMetaCAT class will be kept.

        Args:
            save_path (`str`):
                Path to the json file to load
        '''
        config = cls()

        # Read the jsonpickle string
        with open(save_path) as f:
            config_dict = jsonpickle.decode(f.read())

        config.merge_config(config_dict)

        return config
