from typing import List, Tuple
import torch
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence


class Pad_Sequence():

    def __init__(self, seq_pad_value: int, label_pad_value: int = -1):
        """ Used in rel_cat.py in RelCAT to create DataLoaders for train/test datasets.
            collate_fn for dataloader to collate sequences of different input_ids, ent1/ent2, and label
            lengths into a fixed length batch.
            This is applied per batch and not on the whole DataLoader data,
            padded x sequence, y sequence, x lengths and y lengths of batch.

        Args:
            seq_pad_value (int): pad value for input_ids.
            label_pad_value (int): pad value for labels. Defaults to -1.
        """
        self.seq_pad_value: int = seq_pad_value
        self.label_pad_value: int = label_pad_value

    def __call__(self, batch: List[torch.Tensor]) -> Tuple[Tensor, List, Tensor, LongTensor, LongTensor]:
        """ Pads a batch of input_ids.

        Args:
            batch (List[torch.Tensor]): gets the batch of Tensors from 
                RelData.dataset (check __getitem__() method for data returned)
                and pads the token sequence + labels as needed
                See https://pytorch.org/docs/stable/_modules/torch/nn/utils/rnn.html#pad_sequence 
                for extra info.

        Returns:
            Tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor]: padded data
                padded input ids, ent1&ent2 start token pos, padded labels, padded input_id_lengths, padded labels length
        """
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        # input ids
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(
            seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        # label_ids
        labels = list(map(lambda x: x[2], sorted_batch))
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])

        ent1_ent2_start_pos = list(map(lambda x: x[1], sorted_batch))

        return seqs_padded, ent1_ent2_start_pos, labels_padded, x_lengths, y_lengths
