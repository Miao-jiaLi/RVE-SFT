# REX Construction Pipeline

`build_rex.py` builds the REX training file in one pass from GQA-REX questions, explanations, scene graphs, and object box features.

## Inputs

- `train_balanced_questions_clean.json`
- `converted_explanation_train_balanced.json`
- `train_sceneGraphs.json`
- A box feature directory containing `<image_id>.npy` files

## Usage

```bash
python build_rex.py \
  --questions path/to/train_balanced_questions_clean.json \
  --explanations path/to/converted_explanation_train_balanced.json \
  --box-features path/to/box_features \
  --scene-graphs path/to/train_sceneGraphs.json \
  --output rex_train_4k.json \
  --sample-size 4000 \
  --seed 42
```

The script converts object references into `<ref>...</ref><box>...</box>` evidence tags, filters unresolved references, keeps one sample per image, and writes the final instruction-tuning JSON.

By default, images are saved as file names only, such as `2349751.jpg`. Use `--image-root` only if your training framework requires image paths.
