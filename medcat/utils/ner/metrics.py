from typing import Dict, List
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.special import softmax
import logging

from medcat.cdb import CDB


logger = logging.getLogger(__name__)


def metrics(p, return_df=False, plus_recall=0, tokenizer=None, dataset=None, merged_negative={0, 1, -100}, padding_label=-100, csize=15, subword_label=1,
            verbose=False):
    """
    Calculate metrics for a model's predictions, based off the tokenized output of a MedCATTrainer project.

    Args:
        p: The model's predictions.
        return_df: Whether to return a DataFrame of metrics.
        plus_recall: The recall to add to the model's predictions.
        tokenizer: The tokenizer used to tokenize the texts.
        dataset: The dataset used to train the model.
        merged_negative: The negative labels to merge.
        padding_label: The padding label.
        csize: The size of the context window.
        subword_label: The subword label.
        verbose: Whether to print the metrics.

    """
    """TODO: This could be done better, for sure. But it works."""  # noqa
    predictions = np.array(p.predictions)
    predictions = softmax(predictions, axis=2)
    examples = None
    if plus_recall > 0:
        # Devalue 0 and 1
        predictions[:, :, 0] = predictions[:, :, 0] - (predictions[:, :, 0] * plus_recall)
        predictions[:, :, 1] = predictions[:, :, 1] - (predictions[:, :, 1] * plus_recall)

    preds = np.argmax(predictions, axis=2)
    # Ignore predictions where label == -100, padding
    preds[np.where(p.label_ids == -100)] = -100

    if dataset is not None and tokenizer is not None:
        examples = {'fp': defaultdict(list), 'fn': defaultdict(list)}
        ilbl = {v:k for k,v in tokenizer.label_map.items()}
        for i in range(len(preds)):
            st = None
            for j in range(len(preds[i])):
                _p = preds[i][j]
                _l = p.label_ids[i][j]
                if len(p.label_ids[i]) > (j+1) and p.label_ids[i][j+1] != padding_label:
                    _p2 = preds[i][j+1]
                    _l2 = p.label_ids[i][j+1]
                else:
                    _p2 = None
                    _l2 = None

                _d = dataset[i]['input_ids']
                id = dataset[i]['id']
                name = dataset[i]['name']
                if _l not in {subword_label, padding_label}: # We ignore padding and subwords
                    if _l != _p:
                        if st is None:
                            st = max(0, j-csize)
                            _j = j

                        if not (_l2 is not None and _l2 == _l and _l2 != _p2 and _p2 == _p):
                            # We want to merge tokens if it is the same label and same prediction when recording the examples, that is why we have the if
                            t = tokenizer.hf_tokenizer.decode(_d[st:_j]) + "<<" + str(tokenizer.hf_tokenizer.decode(_d[_j:j+1])) + \
                                ">>" + tokenizer.hf_tokenizer.decode(_d[j+1:j+csize])
                            value = str(tokenizer.hf_tokenizer.decode(_d[_j:j+1])).strip()
                            examples['fp'][ilbl[_p]].append(({'id': id, 'name': name, 'value': value,
                                                              'label': tokenizer.cui2name.get(ilbl[_l], ilbl[_l]), 'text': t}))
                            examples['fn'][ilbl[_l]].append(({'id': id, 'name': name, 'value': value,
                                                              'prediction': tokenizer.cui2name.get(ilbl[_p], ilbl[_p]), 'text': t}))

                            st = None

    _labels = np.reshape(p.label_ids, -1)
    _preds = np.reshape(preds, -1)

    cr = classification_report(_labels, _preds, output_dict=True)
    _cr = {}
    ignore = [str(tokenizer.label_map['O']), str(tokenizer.label_map['X']), '-100']
    ilabel_map = {str(v):k for k,v in tokenizer.label_map.items()}
    for key in cr.keys():
        if key not in ignore and key in ilabel_map:
            _cr[key] = cr[key]

    # Get merged metrics, ie all PID is just one entity
    tp = defaultdict(int)
    fn = defaultdict(int)
    tp_all = 0
    fp_all = 0
    for i, _label in enumerate(_labels):
        _pred = _preds[i]
        if _label in merged_negative:
            if _pred in merged_negative:
                tp[_label] += 1
            else:
                fn[_label] += 1

            if _label == 0:
                if _pred not in merged_negative:
                    fp_all += 1
        else:
            if _pred not in merged_negative:
                tp[_label] += 1
                tp_all += 1
            else:
                fn[_label] += 1
    for key in _cr:
        key = int(key)
        if int(key) in tp:
            _cr[str(key)]['r_merged'] = tp[key] / (tp[key] + fn.get(key, 0)) if tp[key] + fn.get(key, 0) > 0 else 0
        else:
            _cr[str(key)]['r_merged'] = None

    data = [['cui', 'name', 'p', 'r', 'f1', 'support', 'r_merged', 'p_merged']]
    for key in _cr:
        cui = ilabel_map[key]
        p_merged = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
        data.append([cui, tokenizer.cui2name.get(cui, cui), _cr[key]['precision'],
                     _cr[key]['recall'], _cr[key]['f1-score'], _cr[key]['support'], _cr[key]['r_merged'], p_merged])

    df = pd.DataFrame(data[1:], columns=data[0])
    if verbose:
        logger.info('%s', df)

    if not return_df:
        return {'recall': np.average(df.r.values), 'precision': np.average(df.p.values), 'f1': np.average(df.f1.values),
                'recall_merged': np.average([x for x in df.r_merged.values if pd.notna(x)]),
                'precison_merged': np.average([x for x in df.p_merged.values if pd.notna(x)])}
    else:
        return df, examples


def _anno_within_pred_list(label: Dict, preds: List[Dict]) -> bool:
    """
    Check if a label is within a list of predictions,

    Args:
        label (Dict): an annotation likely from a MedCATTrainer project
        preds (List[Dict]): a list of predictions likely from a cat.__call__

    Returns:
        bool: True if the label is within the list of predictions, False otherwise
    """
    return any(label['start'] >= p['start'] and label['end'] <= p['end'] for p in preds)


def evaluate_predictions(true_annotations: List[List[Dict]], all_preds: List[List[Dict]], texts: List[str], deid_cdb: CDB):
    """
    Evaluate predictions against sets of collected labels as collected and output from a MedCATTrainer project.
    Counts predictions as correct if the prediction fully encloses the label.

    Args:
        true_annotations (List[List[Dict]]): Ground truth predictions by text
        all_preds (List[List[Dict]]): Model predictions by text
        texts (List[str]): Original list of texts
        deid_cdb (CDB): Concept database

    Returns:
        Tuple[pd.DataFrame, Dict]: A tuple containing a DataFrame of evaluation metrics and a dictionary of missed annotations per CUI.
    """
    per_cui_recall = {}
    per_cui_prec = {}
    per_cui_recall_merged = {}
    per_cui_anno_counts = {}
    per_cui_annos_missed = defaultdict(list)
    uniq_labels = set([p['cui'] for ap in true_annotations for p in ap])

    for cui in uniq_labels:
        # annos in test set
        anno_count = sum([len([p for p in cui_annos if p['cui'] == cui]) for cui_annos in true_annotations])
        pred_counts = sum([len([p for p in d if p['cui'] == cui]) for d in all_preds])

        # print(anno_count)
        # print(pred_counts)

        # print(f'pred_count: {pred_counts}, anno_count:{anno_count}')
        per_cui_anno_counts[cui] = anno_count

        doc_annos_left, preds_left, doc_annos_left_any_cui = [], [], []

        for doc_preds, doc_labels, text in zip(all_preds, true_annotations, texts):
            # num of annos that are not found - recall
            cui_labels = [label for label in doc_labels if label['cui'] == cui]
            cui_doc_preds = [pred for pred in doc_preds if pred['cui'] == cui]

            labels_not_found = [label for label in cui_labels if not _anno_within_pred_list(label, cui_doc_preds)]
            doc_annos_left.append(len(labels_not_found))

            # num of annos that are not found across any cui prediction - recall_merged
            any_labels_not_found = [label for label in cui_labels if not _anno_within_pred_list(label, doc_preds)]
            doc_annos_left_any_cui.append(len(any_labels_not_found))

            per_cui_annos_missed[cui].append(any_labels_not_found)

            # num of preds that are incorrect - precision
            preds_left.append(len([label for label in cui_doc_preds if not _anno_within_pred_list(label, cui_labels)]))

        if anno_count != 0 and pred_counts != 0:
            per_cui_recall[cui] = (anno_count - sum(doc_annos_left)) / anno_count
            per_cui_recall_merged[cui] = (anno_count - sum(doc_annos_left_any_cui)) / anno_count
            per_cui_prec[cui] = (pred_counts - sum(preds_left))  / pred_counts
        else:
            per_cui_recall[cui] = 0
            per_cui_recall_merged[cui] = 0
            per_cui_prec[cui] = 0

    res_df = pd.DataFrame({
        'cui': per_cui_recall_merged.keys(),
        'recall_merged': per_cui_recall_merged.values(),
        'recall': per_cui_recall.values(),
        'precision': per_cui_prec.values(),
        'label_count': per_cui_anno_counts.values()}, index=[deid_cdb.cui2preferred_name[k] for k in per_cui_recall_merged])

    return res_df, per_cui_annos_missed
