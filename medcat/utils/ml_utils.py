def get_lr_linking(config, cui_count):
    if config.linking['optim']['type'] == 'standard':
        return config.linking['optim']['lr']
    elif config.linking['optim']['type'] == 'linear':
        lr = config.linking['optim']['base_lr']
        cui_count += 1 # Just in case incrase by 1
        return max(lr / cui_count, config.linking['optim']['min_lr'])
    else:
        raise Exception("Optimizer not implemented")


def get_batch(ind, batch_size, x, y, cpos, device):
    # Get the start/end index for this batch
    start = ind * batch_size
    end = (ind+1) * batch_size

    # Get the batch
    x_batch = x[start:end]
    y_batch = y[start:end]
    c_batch = cpos[start:end]

    # Return and move the batches to the right device
    return x_batch.to(device), y_batch.to(device), c_batch.to(device)


def load_hf_tokenizer(tokenizer_name):
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        # Where is log defined?
        log.exception("The Huggingface tokenizer could not be created") # noqa

    return hf_tokenizer


def build_vocab_from_hf(model_name, hf_tokenizer, vocab):
    rebuild = False
    # Check is it necessary
    for i in range(hf_tokenizer.vocab_size):
        tkn = hf_tokenizer.ids_to_tokens[i]
        if tkn not in vocab:
            rebuild = True

    if rebuild:
        # Where is log defined?
        log.info("Rebuilding vocab") # noqa
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
            if 'xlnet' in model_name.lower():
                embs = model.get_input_embeddings().weight.cpu().detach().numpy()
            else:
                embs = model.embeddings.word_embeddings.weight.cpu().detach().numpy()

            # Reset all vecs in current vocab
            vocab.vec_index2word = {}
            for ind in vocab.index2word.keys():
                vocab.vocab[vocab.index2word[ind]]['vec'] = None

            for i in range(hf_tokenizer.vocab_size):
                tkn = hf_tokenizer.ids_to_tokens[i]
                vec = embs[i]
                vocab.add_word(word=tkn, vec=vec, replace=True)

            # Crate the new unigram table
            vocab.reset_counts()
            vocab.make_unigram_table()
        except Exception:
            # Where is log defined?
            log.exception("The Huggingface model could not be loaded") # noqa
