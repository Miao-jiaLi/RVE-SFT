from .language_metrics import lang_eval
from .visual_metrics import vis_eval
from .attribution_metric import attr_evaluater
from .gr_f1_metric import grounding_eval


def evaluation(exp_gts, exp_res):
    scores = lang_eval(exp_gts, exp_res)
    scores['IoU'] = vis_eval(exp_gts, exp_res)
    scores.update(grounding_eval(exp_gts, exp_res))

    return scores
