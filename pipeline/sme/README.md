# SME Construction Pipeline

`build_sme.py` builds SME training or test files in one pass from SME annotations and matching GQA scene graphs.

## Inputs

- `SME_train.json` or `SME_test.json`
- Matching GQA scene graph file, such as `train_sceneGraphs.json` or `val_sceneGraphs.json`

## Usage

Build training data:

```bash
python build_sme.py \
  --input path/to/SME_train.json \
  --scene-graphs path/to/train_sceneGraphs.json \
  --output sme_train_4k.json \
  --mode train \
  --sample-size 4000 \
  --seed 42
```

Build test data for generation:

```bash
python build_sme.py \
  --input path/to/SME_test.json \
  --scene-graphs path/to/val_sceneGraphs.json \
  --output SME_test_lf.json \
  --mode test
```

The script replaces `[BOX]` markers with `<ref>...</ref><box>...</box>` evidence tags. In `train` mode it also keeps one sample per image and writes assistant responses; in `test` mode assistant responses are left empty.

By default, images are saved as file names only, such as `2349751.jpg`. Use `--image-root` only if your training or inference framework requires image paths.
