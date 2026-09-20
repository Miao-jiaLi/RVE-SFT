# Recognizable Visual Evidence Corpus for Fine-Tuning MLLM to Faithful Multimodal Explanatory Visual Question Answering

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

## Baseline Model Prompt Configuration

<div style="text-align: center;">
  <img src="image/prompt.png" alt="Prompts for Baseline model." style="width: 50%; height: auto;">
</div>

The baseline model relies on a structured prompt format that explicitly separates visual entities and their bounding boxes using custom tags such as <ref> and <box>. 
This design forces the model to follow a pre-specified output schema, effectively shifting the grounding burden to prompt engineering rather than enabling the model to acquire grounding behavior from data.
In contrast, our RVE-SFT approach eliminates such structural constraints by training the model to generate grounded explanations end-to-end, where both textual reasoning and coordinate predictions emerge naturally from supervised fine-tuning on the RVE corpus. 
This not only simplifies inference but also enables consistent, verifiable visual grounding without reliance on hand-crafted templates or external tools.

## Showcase of RVE Samples

<div style="text-align: center;">
  <img src="image/sample.png" alt="Fine-tuning corpus RVE samples. Each [BOX] coordinate corresponds to a coloured highlighted bounding box within the image, indicating the match between the annotated text and the visual evidence. Where the term “Visualisation” in grey indicates a visualisation process requiring no visual evidence." style="width: 50%; height: auto;">
</div>

Each sample consists of an image, a question, a natural language explanation with explicit visual grounding, and the corresponding answer.
Explanations are formatted to align entity mentions with normalized bounding box coordinates (x_1,x_2),(y_1,y_2), enabling end-to-end learning of grounded reasoning.
The examples demonstrate the following key characteristics of RVE:

Diverse Visual Scenarios: The corpus covers varied domains, including retail scenes, outdoor activities, street signs, and indoor environments, supporting robust generalization.

Fine-grained Grounding: Grounding targets are not limited to whole objects but can refer to specific parts (e.g., the tie on a shirt, the camera in a hand), enabling precise visual evidence recognition.

Natural Language Integration: Grounding is naturally embedded in fluent explanations, avoiding rigid template-based formats. For instance, the phrase ''the object that the man is holding'' links both semantic meaning and spatial location.

Consistent Format: All entries follow a uniform structure: {"box":"<ref>entity</ref><box>(x_1,x_2),(y_1,y_2)</box>"}, facilitating straightforward supervised fine-tuning without preprocessing.

Visual-Textual Alignment: As shown in the visualization panel, the predicted boxes accurately localize the referenced entities, confirming the effectiveness of the coordinate-aligned supervision.

Overall, these examples illustrate how RVE converts symbolic rationales into machine-readable grounding signals, enabling MLLMs to learn faithful and verifiable explanations through standard supervised training.

## Supplementary Open-world Scene Testing

<div style="text-align: center;">
  <img src="image/Document.png" alt="Document scenario sample." style="width: 50%; height: auto;">
</div>

We evaluate RVE-SFT on a document image containing structured metadata from an academic submission system. 
The task requires extracting a specific date field ("Submitted: 21/Sep/2015") from a complex layout with multiple tables and status indicators.
Qwen3-VL-8B generates a fluent but ungrounded explanation, correctly identifying the date without referencing its visual location. 
In contrast, RVE-SFT explicitly localizes the "submission date" to the region (378,80),(474,106), which precisely matches the text box in the header. 
This demonstrates that RVE-SFT can perform fine-grained visual grounding even in dense, text-heavy scenes—where spatial accuracy is critical for trust and verification.
This capability is particularly valuable in assistive technologies or automated document processing systems, where users need to verify that answers are based on actual content rather than hallucinated information.

<div style="text-align: center;">
  <img src="image/Remote Sensing.png" alt="Remote Sensing scenario sample." style="width: 50%; height: auto;">
</div>

In this scenario, we test RVE-SFT on a satellite image depicting a cityscape with water bodies, roads, and green areas. 
The question requires both semantic understanding and spatial reasoning to determine the location of the largest water body.
Although Qwen3-VL-8B’s answer is plausible, it provides limited evidence. 
However, RVE-SFT grounds its response by referencing two regions: the overall area (0,0),(1000,1000) (the entire image), and the specific water body (308,784),(702,997).
This dual-level grounding shows that RVE-SFT can reason about relative positions and scale, enabling more robust interpretation of geospatial data.
Such ability is essential for applications like environmental monitoring or urban planning, where accurate localization of features from overhead imagery directly impacts decision-making.

<div style="text-align: center;">
  <img src="image/Sport.png" alt="Sport sample." style="width: 50%; height: auto;">
</div>

This example focuses on a real-world sports injury scene, where a person is holding their knee in pain. 
The task is to identify the injured body part from visual cues.
Figure Sport sample shows that both Qwen3-VL-8B and RVE-SFT correctly predict the answer (knee) and provide a plausible explanation. 
Qwen3-VL-8B generates a fluent rationale and outputs two bounding boxes: a coarse box covering the person and a smaller box approximately localizing the knee region. Specifically, the RVE-SFT directly points to the injured area as the red-highlighted region is the knee, offering a clearer semantic explanation. 
This multi-scale grounding reflects the model’s ability to distinguish between object-level and part-level references.
Such a high level of detail is applicable in fields such as triage for sports injuries, where visual evidence aids in diagnosis and reduces misjudgements.
