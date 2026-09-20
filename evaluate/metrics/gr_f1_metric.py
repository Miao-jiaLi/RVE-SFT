def has_grounding(exp):
    return isinstance(exp, str) and "[box]" in exp.lower()


def grounding_eval(exp_gts, exp_res):
    tp = fp = fn = tn = 0
    skipped = 0

    common_qids = exp_gts.keys() & exp_res.keys()
    for qid in common_qids:
        gt_exp = exp_gts[qid].get("explanation")
        pred_exp = exp_res[qid].get("explanation")

        if not isinstance(gt_exp, str):
            skipped += 1
            continue

        gt = has_grounding(gt_exp)
        pred = has_grounding(pred_exp)

        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "Grounding_TP": tp,
        "Grounding_FP": fp,
        "Grounding_FN": fn,
        "Grounding_TN": tn,
        "Grounding_Precision": precision * 100,
        "Grounding_Recall": recall * 100,
        "GR-F1": f1 * 100,
        "Grounding_Skipped": skipped,
        "Grounding_Total": tp + fp + fn + tn,
    }
