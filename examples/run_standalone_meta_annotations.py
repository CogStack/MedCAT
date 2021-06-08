from medcat.datasets.helpers import encode_examples
from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from medcat.meta_cat import MetaCAT
import torch

CACHE_DIR = './'
DATASETS_CLASS_PATH = "../medcat/datasets/medcat_annotations.py"
ANNOTATIONS_PATH = ''
TOKENIZER_PATH = ''
MAX_SEQ_LEN = 100
BATCH_SIZE = 1024
DEVICE = torch.device('cuda')
META_CAT_SAVE_DIR = ''

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

dataset = load_dataset(DATASETS_CLASS_PATH, data_files=ANNOTATIONS_PATH, name='pickle', split='train', cache_dir=CACHE_DIR)

encoded_dataset = dataset.map(
        lambda examples: encode_examples(examples['context_left'], examples['context_center'], examples['context_right'], tokenizer, MAX_SEQ_LEN),
        batched=True,
        remove_columns=['context_center', 'context_left', 'context_right', 'document_id', 'id'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=MAX_SEQ_LEN)

dataset_loader = DataLoader(encoded_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

meta_cat = MetaCAT(save_dir=META_CAT_SAVE_DIR, tokenizer=tokenizer)
meta_cat.load()
meta_cat.model.eval()
meta_cat.model.to(DEVICE)

logits = []
for i, batch in enumerate(dataset_loader):
    batch = {k:v.to(DEVICE) for k,v in batch.items()}
    out = meta_cat.model(**batch)
    logits.extend(out.to('cpu').detach().numpy().tolist())

    if i % 1000 == 0:
        print("Done with batch: {}".format(i))

predictions = np.argmax(logits, axis=1)
