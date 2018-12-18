from preprocessing.vocab import Vocab


vocab = Vocab()

# First add words that have vectors
vocab.add_words('/home/ubuntu/data/other/dicts/wiki-umls-emb.txt')

# Now add words without vectors
vocab.add_words_nvec('/home/ubuntu/data/other/dicts/wiki-umls.txt')


vocab.save("/home/ubuntu/data/other/dicts/umls-wiki-emb.dat")

