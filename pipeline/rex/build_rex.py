import argparse
import json
import random
import re
from pathlib import Path


OBJECT_REF_PATTERN = re.compile(r"#(\d+)")
BOX_PATTERN = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
DEFAULT_IOU_THRESHOLD = 0.7


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_box_coordinates(box_features_dir, image_id, object_id):
    import numpy as np

    box_path = Path(box_features_dir) / f"{image_id}.npy"
    if not box_path.exists():
        return None

    boxes = np.load(box_path)
    if object_id < 0 or object_id >= len(boxes):
        return None

    return [int(coord) for coord in boxes[object_id]]


def replace_object_refs_with_boxes(explanation, image_id, box_features_dir):
    def replace(match):
        object_id = int(match.group(1))
        box = load_box_coordinates(box_features_dir, image_id, object_id)
        return str(box) if box is not None else match.group(0)

    return OBJECT_REF_PATTERN.sub(replace, explanation)


def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_width = max(0, x_b - x_a)
    inter_height = max(0, y_b - y_a)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def normalize_box(box, width, height):
    x1, y1, x2, y2 = box
    normalized = [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]
    return [max(0, min(value, 1000)) for value in normalized]


def find_best_object(box, scene_graph_objects):
    best_object = None
    best_score = 0.0

    for obj in scene_graph_objects.values():
        obj_box = [obj["x"], obj["y"], obj["x"] + obj["w"], obj["y"] + obj["h"]]
        score = compute_iou(box, obj_box)
        if score > best_score:
            best_score = score
            best_object = obj

    return best_object, best_score


def format_ref_box(box, obj, width, height):
    x1, y1, x2, y2 = normalize_box(box, width, height)
    label = obj["name"] if obj is not None else "unknown"
    return f"<ref>{label}</ref><box>({x1},{y1}),({x2},{y2})</box>"


def parse_boxes(text):
    return [
        [int(match.group(i)) for i in range(1, 5)]
        for match in BOX_PATTERN.finditer(text)
    ]


def select_valid_samples(questions, explanations):
    samples = {}
    for qid, item in questions.items():
        explanation = explanations.get(qid, "")
        if not explanation or "@" in explanation:
            continue

        samples[qid] = {
            "qid": qid,
            "question": item["question"],
            "answer": item["answer"],
            "imageId": item["imageId"],
            "explanation": explanation,
        }
    return samples


def add_raw_boxes(samples, box_features_dir):
    output = {}
    for qid, item in samples.items():
        image_id = item["imageId"]
        original_explanation = item["explanation"]
        output[qid] = {
            "qid": item["qid"],
            "question": item["question"],
            "answer": item["answer"],
            "imageId": image_id,
            "original_explanation": original_explanation,
            "converted_explanation": replace_object_refs_with_boxes(
                original_explanation, image_id, box_features_dir
            ),
        }
    return output


def add_ref_box_tags(samples, scene_graphs, iou_threshold):
    output = {}
    missing_scene_graphs = 0

    for qid, item in samples.items():
        image_id = item["imageId"]
        if image_id not in scene_graphs:
            missing_scene_graphs += 1
            continue

        scene_graph = scene_graphs[image_id]
        width = scene_graph["width"]
        height = scene_graph.get("height", 500)
        objects = scene_graph["objects"]

        labels = []
        for box in parse_boxes(item["converted_explanation"]):
            obj, score = find_best_object(box, objects)
            if obj is None or score < iou_threshold:
                obj = None
            labels.append(format_ref_box(box, obj, width, height))

        label_iter = iter(labels)
        full_explanation = BOX_PATTERN.sub(lambda _: next(label_iter), item["converted_explanation"])
        output[qid] = {
            **item,
            "full_explanation": full_explanation.rstrip(".") + ".",
        }

    return output, missing_scene_graphs


def remove_unknown_refs(samples):
    return {
        qid: sample
        for qid, sample in samples.items()
        if "unknown" not in sample.get("full_explanation", "").lower()
    }


def keep_one_sample_per_image(samples, rng):
    image_to_samples = {}
    for qid, sample in samples.items():
        image_to_samples.setdefault(sample["imageId"], []).append((qid, sample))

    output = {}
    for grouped_samples in image_to_samples.values():
        qid, sample = rng.choice(grouped_samples)
        output[qid] = sample
    return output


def build_image_path(image_id, image_root):
    image_name = f"{image_id}.jpg"
    if not image_root:
        return image_name
    return f"{image_root.rstrip('/')}/{image_name}"


def to_instruction_tuning_format(samples, rng, sample_size, image_root):
    data = list(samples.values())
    if sample_size is not None:
        data = rng.sample(data, min(sample_size, len(data)))

    output = []
    for item in data:
        question = item["question"]
        answer = item["answer"]
        explanation = item["full_explanation"]
        image_path = build_image_path(item["imageId"], image_root)

        output.append(
            {
                "images": [image_path],
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<image>Please provide a grounded explanation using the image, "
                            "and answer the question based on this explanation.\n"
                            f"Question: {question}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": f"Explanation: {explanation}\nAnswer: {answer}",
                    },
                ],
            }
        )
    return output


def build_rex_dataset(args):
    rng = random.Random(args.seed)
    questions = load_json(args.questions)
    explanations = load_json(args.explanations)
    scene_graphs = load_json(args.scene_graphs)

    raw_samples = select_valid_samples(questions, explanations)
    boxed_samples = add_raw_boxes(raw_samples, args.box_features)
    ref_samples, missing_scene_graphs = add_ref_box_tags(
        boxed_samples, scene_graphs, DEFAULT_IOU_THRESHOLD
    )
    clean_samples = remove_unknown_refs(ref_samples)
    clean_samples = keep_one_sample_per_image(clean_samples, rng)

    final_data = to_instruction_tuning_format(
        clean_samples, rng, args.sample_size, args.image_root
    )
    save_json(final_data, args.output)

    print(f"Valid samples: {len(raw_samples)}")
    print(f"Samples after ref-box conversion: {len(ref_samples)}")
    print(f"Samples skipped for missing scene graphs: {missing_scene_graphs}")
    print(f"Samples after unknown filtering: {len(clean_samples)}")
    print(f"Final samples: {len(final_data)}")
    print(f"Saved to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build REX instruction-tuning data in one pass.")
    parser.add_argument("--questions", required=True, help="Path to the GQA-REX question JSON.")
    parser.add_argument("--explanations", required=True, help="Path to the GQA-REX explanation JSON.")
    parser.add_argument("--scene-graphs", required=True, help="Path to the GQA scene graph JSON.")
    parser.add_argument("--box-features", required=True, help="Directory containing <image_id>.npy box files.")
    parser.add_argument("--output", required=True, help="Path to the final instruction-tuning JSON.")
    parser.add_argument("--sample-size", type=int, default=4000, help="Number of final samples to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--image-root", default="", help="Optional image root. Empty means image file names only.")
    return parser.parse_args()


def main():
    build_rex_dataset(parse_args())


if __name__ == "__main__":
    main()
