from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge


def compute_scores(gts, res):
    """
    Performs evaluation using only BLEU and ROUGE metrics without Java dependency
    """
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(), "ROUGE_L")
    ]
    
    eval_res = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    
    return eval_res


def compute_mlc(gt, pred, label_set):
    res_mlc = {}
    avg_aucroc = 0
    for i, label in enumerate(label_set):
        res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
        avg_aucroc += res_mlc['AUCROC_' + label]
    res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)

    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc
