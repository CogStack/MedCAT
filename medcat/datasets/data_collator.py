from typing import Any, Dict, List
import torch


class CollateAndPadNER(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features: List[Any]) -> Dict[str, torch.Tensor]:
        batch = {}

        max_len = max([len(f['input_ids']) for f in features])
        batch['input_ids'] = torch.tensor([f['input_ids'][0:max_len] + [self.pad_id] *
            max(0, max_len - len(f['input_ids'])) for f in features], dtype=torch.long)
        batch['labels'] = torch.tensor([f['labels'][0:max_len] + [-100] *
            max(0, max_len - len(f['labels'])) for f in features], dtype=torch.long)
        batch['attention_mask'] = batch['input_ids'] != self.pad_id

        return batch
