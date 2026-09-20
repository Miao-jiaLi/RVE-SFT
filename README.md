# RVE: A Data-centric Recognizable Visual Evidence Corpus for Multimodal Explanatory Visual Question Answering

This repository contains the released data and evaluation code for the paper of Recognizable Visual Evidence Corpus for Fine-Tuning MLLM to Faithful Multimodal Explanatory Visual Question Answering. 
It includes supervised fine-tuning data with recognizable visual evidence annotations and an evaluation toolkit for measuring explanation quality, visual grounding, attribution, and answer accuracy.


## Dataset

The `dataset/` directory provides two training files:

- `sme_train_4k.json`: the processed 4K training set.
- `sme_train_4k_raw_exp.json`: the corresponding 4K training set with raw explanations.

The associated image files should be placed or resolved according to these file names in the user's local data environment.

### Downloading Source Data

The image files are not redistributed in this repository. They can be downloaded from the official GQA image source:

- GQA official download page: https://cs.stanford.edu/people/dorarad/gqa/download.html

The full SME dataset annotations are publicly available from the original release:

- Hugging Face: https://huggingface.co/datasets/LivXue/Standard-Multimodal-Explanation
- GitHub: https://github.com/LivXue/FS-MEVQA

After downloading GQA images, keep them in a directory where each image can be addressed by its file name, such as `2349751.jpg`, matching the `images` field in the released JSON files.

## Construction Pipeline

The released data construction notes and cleaned scripts are provided under `pipeline/`.

- `pipeline/rex/`: REX construction pipeline, including the cleaned `build_rex.py` command-line workflow.
- `pipeline/sme/`: SME construction pipeline, including the cleaned `build_sme.py` command-line workflow.

## Evaluation

The `evaluate/` directory contains the evaluation entry point and metric implementations.

Run evaluation with:

```bash
python evaluate/evaluation.py --pred PATH_TO_PREDICTIONS.json --gt PATH_TO_GROUND_TRUTH.json
```

The prediction and ground-truth files are expected to be JSON dictionaries keyed by sample ID. Each entry should include at least:

- `answer`: the final answer.
- `explanation`: the generated or reference explanation.
- `boxes`: grounding boxes used by visual grounding metrics.

The evaluation script reports answer accuracy and explanation metrics, including:

- Language metrics from `language_metrics.py`.
- Visual grounding IoU from `visual_metrics.py`.
- Attribution scores from `attribution_metric.py`.
- `GR-F1` from `gr_f1_metric.py`, which measures whether predicted explanations correctly include recognizable visual evidence markers when the reference explanation requires them.

## Experimental Framework

The released instruction-tuning files follow a LLaMA-Factory-compatible multimodal data format. Experiments can be conducted with LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory


