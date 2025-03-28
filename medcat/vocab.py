import numpy as np
import pickle
from typing import Optional, List, Dict, cast
import logging


logger = logging.getLogger(__name__)


class Vocab(object):
    """Vocabulary used to store word embeddings for context similarity
    calculation. Also used by the spell checker - but not for fixing the spelling
    only for checking is something correct.

    Properties:
        vocab (dict):
            Map from word to attributes, e.g. {'house': {'vec': <np.array>, 'cnt': <int>, ...}, ...}
        index2word (dict):
            From word to an index - used for negative sampling
        vec_index2word (dict):
            Same as index2word but only words that have vectors
        unigram_table (dict):
            Negative sampling.
    """
    def __init__(self) -> None:
        self.vocab: Dict = {}
        self.index2word: Dict = {}
        self.vec_index2word: Dict = {}
        self.cum_probs = np.array([])

    def inc_or_add(self, word: str, cnt: int = 1, vec: Optional[np.ndarray] = None) -> None:
        """Add a word or increase its count.

        Args:
            word(str):
                Word to be added
            cnt(int):
                By how much should the count be increased, or to what
                should it be set if a new word. (Default value = 1)
            vec(Optional[np.ndarray]):
                Word vector (Default value = None)
        """
        if word not in self.vocab:
            self.add_word(word, cnt, vec)
        else:
            self.inc_wc(word, cnt)

    def remove_all_vectors(self) -> None:
        """Remove all stored vector representations."""
        self.vec_index2word = {}

        for word in self.vocab:
            self.vocab[word]['vec'] = None

    def remove_words_below_cnt(self, cnt: int) -> None:
        """Remove all words with frequency below cnt.

        Args:
            cnt(int):
                Word count limit.
        """
        for word in list(self.vocab.keys()):
            if self.vocab[word]['cnt'] < cnt:
                del self.vocab[word]

        # Rebuild index2word and vec_index2word
        self.index2word = {}
        self.vec_index2word = {}
        for word in self.vocab.keys():
            ind = len(self.index2word)
            self.index2word[ind] = word
            self.vocab[word]['ind'] = ind

            if self.vocab[word]['vec'] is not None:
                self.vec_index2word[ind] = word

    def inc_wc(self, word: str, cnt: int = 1) -> None:
        """Incraese word count by cnt.

        Args:
            word(str):
                For which word to increase the count
            cnt(int):
                By how muhc to increase the count (Default value = 1)
        """
        self.item(word)['cnt'] += cnt

    def add_vec(self, word: str, vec: np.ndarray) -> None:
        """Add vector to a word.

        Args:
            word(str):
                To which word to add the vector.
            vec(np.ndarray):
                The vector to add.
        """
        self.vocab[word]['vec'] = vec

        ind = self.vocab[word]['ind']
        if ind not in self.vec_index2word:
            self.vec_index2word[ind] = word

    def reset_counts(self, cnt: int = 1) -> None:
        """Reset the count for all word to cnt.

        Args:
            cnt(int):
                New count for all words in the vocab. (Default value = 1)
        """
        for word in self.vocab.keys():
            self.vocab[word]['cnt'] = cnt

    def update_counts(self, tokens: List[str]) -> None:
        """Given a list of tokens update counts for words in the vocab.

        Args:
            tokens(List[str]):
                Usually a large block of text split into tokens/words.
        """
        for token in tokens:
            if token in self:
                self.inc_wc(token, 1)

    def add_word(self, word: str, cnt: int = 1, vec: Optional[np.ndarray] = None, replace: bool = True) -> None:
        """Add a word to the vocabulary

        Args:
            word(str):
                The word to be added, it should be lemmatized and lowercased
            cnt(int):
                Count of this word in your dataset (Default value = 1)
            vec(Optional[np.ndarray]):
                The vector representation of the word (Default value = None)
            replace(bool):
                Will replace old vector representation (Default value = True)
        """
        if word not in self.vocab:
            ind = len(self.index2word)
            self.index2word[ind] = word
            item = {'vec': vec, 'cnt': cnt, 'ind': ind}
            self.vocab[word] = item

            if vec is not None:
                self.vec_index2word[ind] = word
        elif replace and vec is not None:
            self.vocab[word]['vec'] = vec
            self.vocab[word]['cnt'] = cnt

            # If this word didn't have a vector before
            ind = self.vocab[word]['ind']
            if ind not in self.vec_index2word:
                self.vec_index2word[ind] = word

    def add_words(self, path: str, replace: bool = True) -> None:
        """Adds words to the vocab from a file, the file
        is required to have the following format (vec being optional):
            <word>\t<cnt>[\t<vec_space_separated>]

        e.g. one line: the word house with 3 dimensional vectors
            house   34444   0.3232 0.123213 1.231231

        Args:
            path(str):
                path to the file with words and vectors
            replace(bool):
                existing words in the vocabulary will be replaced (Default value = True)
        """
        with open(path) as f:
            for line in f:
                parts = line.split("\t")
                word = parts[0]
                cnt = int(parts[1].strip())
                vec = None
                if len(parts) == 3:
                    vec = np.array([float(x) for x in parts[2].strip().split(" ")])

                self.add_word(word, cnt, vec, replace)

    def make_unigram_table(self, table_size: int = -1) -> None:
        """Make unigram table for negative sampling, look at the paper if interested
        in details.

        Args:
            table_size (int):
                The size of the table - no longer needed (Defaults to -1)
        """
        if table_size != -1:
            logger.warning("Unigram table size is no longer necessary since "
                           "there is now a simpler approach that doesn't require "
                           "the creation of a massive array. So therefore, there "
                           "is no need to pass the `table_size` parameter anymore.")
        freqs = []
        # index list maps the slot in which a word index
        # sits in vec_index2word to the actual index for said word
        # e.g:
        #    if we have words indexed 0, 1, and 2
        #    but only 0, and 2 have corresponding vectors
        #    then only 0 and 2 will occur in vec_index2word
        #    and while 0 will be in the 0th position (as expected)
        #    in the final probability list, 2 will be in 1st position
        #    so we need to mark that conversion down
        index_list = []
        for word_index, word in self.vec_index2word.items():
            freqs.append(self[word])
            index_list.append(word_index)

        # Power and normalize frequencies
        freqs = np.array(freqs) ** (3/4)
        freqs /= freqs.sum()

        # Calculate cumulative probabilities
        self.cum_probs = np.cumsum(freqs)
        # the mapping from vector index order to word indices
        self._index_list = index_list

    def get_negative_samples(self, n: int = 6, ignore_punct_and_num: bool = False) -> List[int]:
        """Get N negative samples.

        Args:
            n (int):
                How many words to return (Default value = 6)
            ignore_punct_and_num (bool):
                Whether to ignore punctuation and numbers. (Default value = False)

        Returns:
            List[int]:
                Indices for words in this vocabulary.
        """
        if len(self.cum_probs) == 0:
            self.make_unigram_table()
        random_vals = np.random.rand(n)
        # NOTE: These indices are in terms of the cum_probs array
        #       which only has word data for words with vectors.
        vec_slots = cast(List[int], np.searchsorted(self.cum_probs, random_vals).tolist())
        # so we need to translate these back to word indices
        inds = list(map(self._index_list.__getitem__, vec_slots))

        if ignore_punct_and_num:
            # Do not return anything that does not have letters in it
            inds = [ind for ind in inds if self.index2word[ind].upper().isupper()]

        return inds

    def __getitem__(self, word: str) -> int:
        return self.count(word)

    def vec(self, word: str) -> np.ndarray:
        return self.vocab[word]['vec']

    def count(self, word: str) -> int:
        return self.vocab[word]['cnt']

    def item(self, word: str) -> Dict:
        return self.vocab[word]

    def __contains__(self, word: str) -> bool:
        if word in self.vocab:
            return True

        return False

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, 'rb') as f:
            vocab = cls()
            vocab.__dict__ = pickle.load(f)
        if not hasattr(vocab, 'cum_probs'):
            # NOTE: this is not too expensive, only around 0.05s
            vocab.make_unigram_table()
        if hasattr(vocab, 'unigram_table'):
            del vocab.unigram_table
        return vocab
