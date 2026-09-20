import argparse
import json
import random
import re
from pathlib import Path


BOX_TOKEN = "[BOX]"
DIRECT_PROMPT = (
    "<image>Please provide a grounded explanation using the image, and answer "
    "the question based on this explanation.\nQuestion: {question}"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_box(box, width, height):
    x1, y1, x2, y2 = box
    normalized = [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]
    return [max(0, min(value, 1000)) for value in normalized]


def format_ref_box(label, box, width, height):
    x1, y1, x2, y2 = normalize_box(box, width, height)
    return f"<ref>{label}</ref><box>({x1},{y1}),({x2},{y2})</box>"


def replace_next_box(text, label, ref_box):
    pattern = rf"\b{re.escape(label)}\s*\[BOX\]"
    match = re.search(pattern, text)
    if match:
        return text[: match.start()] + ref_box + text[match.end() :]
    return text.replace(BOX_TOKEN, ref_box, 1)


def iter_ordered_boxes(boxes):
    for label, box_groups in boxes.items():
        for group in box_groups:
            for box in group:
                yield label, box


def add_ref_box_tags(samples, scene_graphs):
    output = {}
    missing_scene_graphs = 0

    for qid, item in samples.items():
        image_id = item["imageId"]
        if image_id not in scene_graphs:
            missing_scene_graphs += 1
            continue

        scene_graph = scene_graphs[image_id]
        width = scene_graph["width"]
        height = scene_graph["height"]
        explanation = item["explanation"]

        for label, box in iter_ordered_boxes(item.get("boxes", {})):
            ref_box = format_ref_box(label, box, width, height)
            explanation = replace_next_box(explanation, label, ref_box)

        output[qid] = {
            **{k: v for k, v in item.items() if k != "boxes"},
            "qid": qid,
            "full_explanation": explanation,
        }

    return output, missing_scene_graphs


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


def make_training_record(item, image_root):
    question = item["question"]
    answer = item["answer"]
    explanation = item["full_explanation"]
    return {
        "images": [build_image_path(item["imageId"], image_root)],
        "messages": [
            {"role": "user", "content": DIRECT_PROMPT.format(question=question)},
            {"role": "assistant", "content": f"Explanation: {explanation}\nAnswer: {answer}"},
        ],
    }


def make_test_record(item, image_root):
    question = item["question"]

    return {
        "images": [build_image_path(item["imageId"], image_root)],
        "messages": [
            {"role": "user", "content": DIRECT_PROMPT.format(question=question)},
            {"role": "assistant", "content": ""},
        ],
    }


def to_output_format(samples, rng, mode, sample_size, image_root):
    data = list(samples.values())
    if sample_size is not None:
        data = rng.sample(data, min(sample_size, len(data)))

    if mode == "train":
        return [make_training_record(item, image_root) for item in data]
    return [make_test_record(item, image_root) for item in data]


def build_sme_dataset(args):
    rng = random.Random(args.seed)
    samples = load_json(args.input)
    scene_graphs = load_json(args.scene_graphs)

    processed_samples, missing_scene_graphs = add_ref_box_tags(samples, scene_graphs)
    if args.mode == "train":
        processed_samples = keep_one_sample_per_image(processed_samples, rng)

    final_data = to_output_format(
        processed_samples, rng, args.mode, args.sample_size, args.image_root
    )
    save_json(final_data, args.output)

    print(f"Input samples: {len(samples)}")
    print(f"Samples skipped for missing scene graphs: {missing_scene_graphs}")
    print(f"Processed samples: {len(processed_samples)}")
    print(f"Final samples: {len(final_data)}")
    print(f"Saved to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build SME data in one pass.")
    parser.add_argument("--input", required=True, help="Path to the SME annotation JSON.")
    parser.add_argument("--scene-graphs", required=True, help="Path to the matching GQA scene graph JSON.")
    parser.add_argument("--output", required=True, help="Path to the final JSON file.")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Final output type to build.",
    )
    parser.add_argument("--sample-size", type=int, default=4000, help="Number of samples to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--image-root", default="", help="Optional image root. Empty means image file names only.")
    return parser.parse_args()


def main():
    build_sme_dataset(parse_args())


if __name__ == "__main__":
    main()
