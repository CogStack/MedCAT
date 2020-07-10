from sklearn.model_selection import train_test_split
import numpy as np
from medcat.utils.models import LSTM as MODEL
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from medcat.utils.loggers import basic_logger
log = basic_logger("utils")


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


def train_network(net, data, lr=0.01, test_size=0.1, max_seq_len=41, pad_id=30000, batch_size=100,
                  nepochs=20, device='cpu', save_dir='./meta_cat/', class_weights=None, ignore_cpos=False,
                  auto_save_model=True, score_average='weighted'):
    # Split data
    y = np.array([x[0] for x in data])
    x = [x[1] for x in data]
    cent = np.array([x[2] for x in data])

    # Pad X and convert to array
    x = np.array([(sample + [pad_id] * max(0, max_seq_len - len(sample)))[0:max_seq_len] for sample in x])

    x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(x, y,
            cent, test_size=test_size)

    x_train = torch.tensor(x_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    c_train = torch.tensor(c_train, dtype=torch.long)

    x_test = torch.tensor(x_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    c_test = torch.tensor(c_test, dtype=torch.long)

    device = torch.device(device) # Create a torch device
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights) # Set the criterion to Cross Entropy Loss
    else:
        criterion = nn.CrossEntropyLoss() # Set the criterion to Cross Entropy Loss
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=lr)
    net.to(device) # Move the network to device

    batch_size = 40
    # Calculate the number of batches given training size len(x_train)
    num_batches = int(np.ceil(len(x_train) / batch_size))
    best_f1 = 0
    best_p = 0
    best_r = 0
    best_cls_report = None
    for epoch in range(nepochs):
        running_loss_train = []
        running_loss_test = []

        # TRAIN PHASE
        net.train()
        train_outs = []
        for i in range(num_batches):
            x_train_batch, y_train_batch, cpos_train_batch = get_batch(ind=i,
                                                                       batch_size=batch_size,
                                                                       x=x_train,
                                                                       y=y_train, cpos=c_train,
                                                                       device=device)
            optimizer.zero_grad()
            outputs = net(x_train_batch, cpos_train_batch, ignore_cpos=ignore_cpos)
            loss = criterion(outputs, y_train_batch)
            loss.backward()
            running_loss_train.append(loss.item())

            parameters = filter(lambda p: p.requires_grad, net.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 0.25)
            optimizer.step()

            train_outs.append(outputs.detach().cpu().numpy())


        # TEST PHASE

        num_batches_test = int(np.ceil(len(x_test) / batch_size))
        test_outs = []
        for j in range(num_batches_test):
            net.eval()
            x_test_batch, y_test_batch, cpos_test_batch = get_batch(ind=j,
                                                                     batch_size=batch_size,
                                                                     x=x_test,
                                                                     y=y_test,
                                                                     cpos=c_test,
                                                                     device=device)
            outputs_test = net(x_test_batch, cpos_test_batch, ignore_cpos=ignore_cpos)
            test_outs.append(outputs_test.detach().cpu().numpy())
            running_loss_test.append(criterion(outputs_test, y_test_batch).item())

        train_loss = np.average(running_loss_train)
        test_loss = np.average(running_loss_test)

        print("*"*50 + "  Train")
        print(classification_report(y_train, np.argmax(np.concatenate(train_outs, axis=0), axis=1)))
        print("*"*50 + "  Test")
        print(classification_report(y_test, np.argmax(np.concatenate(test_outs, axis=0), axis=1)))
        print("Train Loss: {:5}\nTest Loss:  {:5}\n\n".format(train_loss, test_loss))
        print("\n\n\n")
        f1 = f1_score(y_test, np.argmax(np.concatenate(test_outs, axis=0), axis=1), average=score_average)
        precision = precision_score(y_test, np.argmax(np.concatenate(test_outs, axis=0), axis=1), average=score_average)
        recall = recall_score(y_test, np.argmax(np.concatenate(test_outs, axis=0), axis=1), average=score_average)
        cls_report = classification_report(y_test, np.argmax(np.concatenate(test_outs, axis=0), axis=1), output_dict=True)

        if f1 > best_f1:
            print("=" * 50)
            if auto_save_model:
                path = save_dir + "lstm.dat"
                torch.save(net.state_dict(), path)
                print("Model saved at epoch: {} and f1: {}".format(epoch, f1))

            best_f1 = f1
            best_p = precision
            best_r = recall
            best_cls_report = cls_report

            # Print some stats
            print(confusion_matrix(y_test, np.argmax(np.concatenate(test_outs, axis=0), axis=1)))
            print("\n\n")

    return (best_f1, best_p, best_r, best_cls_report)


def load_hf_tokenizer(tokenizer_name):
    try:
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        log.exception("The Huggingface tokenizer could not be created")

    return hf_tokenizer


def build_vocab_from_hf(model_name, hf_tokenizer, vocab):
    rebuild = False
    # Check is it necessary
    for i in range(hf_tokenizer.vocab_size):
        tkn = hf_tokenizer.ids_to_tokens[i]
        if tkn not in vocab:
            rebuild = True

    if rebuild:
        log.info("Rebuilding vocab")
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
        except Exception as e:
            log.exception("The Huggingface model could not be loaded")
