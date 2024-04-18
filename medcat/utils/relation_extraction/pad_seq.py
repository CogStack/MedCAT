from typing import List, Tuple
import torch
from torch import Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence


class Pad_Sequence():

    def __init__(self, seq_pad_value: int, label_pad_value: int = -1, label2_pad_value: int = -1,
                 ):
        """ Used in rel_cat.py in RelCAT to create DataLoaders for train/test datasets.
            collate_fn for dataloader to collate sequences of different lengths into a fixed length batch

        Args:
            padded x sequence, y sequence, x lengths and y lengths of batch
            seq_pad_value (int): _description_
            label_pad_value (int, optional): _description_. Defaults to -1.
            label2_pad_value (int, optional): _description_. Defaults to -1.
        """
        self.seq_pad_value: int = seq_pad_value
        self.label_pad_value: int = label_pad_value
        self.label2_pad_value: int = label2_pad_value

    def __call__(self, batch: List[torch.Tensor]) -> Tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor, LongTensor]:
        """

        Args:
            batch (List[torch.Tensor]): gets the batch of Tensors from RelData(Dataset) and
            pads the token sequence + labels as needed

        Returns:
            Tuple[Tensor, Tensor, Tensor, LongTensor, LongTensor, LongTensor]: padded data
        """
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(
            seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])

        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])

        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(
            labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])

        return seqs_padded, labels_padded, labels2_padded, \
            x_lengths, y_lengths, y2_lengths
