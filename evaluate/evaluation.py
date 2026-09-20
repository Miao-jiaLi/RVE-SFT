import argparse
import json


class Evaluator:
    def __init__(self, gts):
        from metrics import attr_evaluater

        self.exp_gts = gts
        self.attr_evaluater = attr_evaluater(gts)

    def evaluate(self, exp_res):
        from metrics import grounding_eval, lang_eval, vis_eval

        scores = lang_eval(self.exp_gts, exp_res)
        scores['IoU'] = vis_eval(self.exp_gts, exp_res)
        scores.update(self.attr_evaluater.score(exp_res))
        scores.update(grounding_eval(self.exp_gts, exp_res))

        return scores


evaluator = Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SME prediction results.")
    parser.add_argument("--pred", required=True, help="Path to the prediction result JSON file.")
    parser.add_argument("--gt", required=True, help="Path to the ground-truth JSON file.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def main():
    args = parse_args()
    res = load_json(args.pred)
    gts = load_json(args.gt)
    gts = {k: gts[k] for k in res}

    print(len(gts))
    assert res.keys() == gts.keys()

    evaluator = Evaluator(gts)

    acc = []
    for k in res.keys():
        if res[k]['answer'] == gts[k]['answer']:
            acc.append(1)
        else:
            acc.append(0)

    acc = sum(acc) / len(acc)
    print(f"Accuracy = {acc * 100}")

    scores = evaluator.evaluate(res)
    for k in scores:
        print(f"{k} = {scores[k]}")


if __name__ == '__main__':
    main()
