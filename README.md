# Multimodal Clickbait Detection

Detecting clickbait news articles from their **title, description, and thumbnail image**. The model combines three encoders, a text encoder with learnable soft prompts, an image encoder, and a syntactic graph encoder over the title, and fuses them for binary classification. A weighted ensemble of three trained variants gives the final prediction.

Course project for Natural Language Processing at HUFLIT (2025), trained on a Google Colab A100.

## Results

Test set: 1,706 articles (856 non-clickbait, 850 clickbait).

Weighted ensemble of the three variants with the highest validation accuracy:

| Accuracy | F1 | Recall | PR-AUC |
| --- | --- | --- | --- |
| **82.83%** | **82.73%** | 82.59% | 0.907 |

## Dataset

- **Source:** English-language news articles from Irish and UK publishers (e.g. thesun.ie, irishmirror.ie, irishexaminer.com, bbc.com), each annotated with a `clickbait_level` from 1 to 5.
- **Labels:** level 1 is *not clickbait* (0), levels 4–5 are *clickbait* (1). Levels 2–3 are left out to keep the two classes clearly separated.
- **Balancing:** all 8,613 level 4–5 articles are kept and the same number of level-1 articles is sampled at random. After dropping rows with a missing title, description, or thumbnail, **17,051 articles** remain (8,551 / 8,500).
- **Thumbnails:** crawled from each article page (`og:image`, falling back to the first `<img>`), then resized to 200×200.
- **Split:** stratified 70 / 20 / 10 → train 11,935 · validation 3,410 · test 1,706.

## Model

```
 title + description ──► RoBERTa-large  (+5 soft prompt tokens) ──► 256-d ─┐
 thumbnail           ──► CLIP ViT-B/32  (+soft prompt, conv head) ──► 256-d ─┼─► fusion ──► clickbait logit
 title               ──► spaCy parse ──► GAT over dependency graph ──► 256-d ─┘
```

1. **Text encoder:** `roberta-large` over the title and description (max 128 tokens). Five learnable prompt vectors are prepended to the input embeddings, and the output is mean-pooled over prompt and text tokens.
2. **Image encoder:** CLIP `clip-vit-base-patch32` vision tower. A learnable prompt vector is added to the patch-token sequence, followed by a 1×1 convolution, average pooling, and a linear projection.
3. **Syntactic encoder:** the title is parsed with spaCy (`en_core_web_sm`). Fine-grained POS tags are embedded as node features, dependency arcs are the edges, and a Graph Attention layer (`GATConv`) encodes the graph, which is then mean-pooled.
4. **Fusion and head:** the three 256-d vectors are concatenated and passed to either
   - **concat:** an MLP classifier, or
   - **gated:** a sigmoid gate that re-weights each feature before the MLP.

All encoders are fine-tuned end-to-end for 3 epochs with AdamW, using either **focal loss** or **class-balanced BCE**.

**Ensemble:** each model votes with its thresholded prediction, weighted by a softmax over its validation accuracy (temperature 0.3). PR-AUC uses the weighted average of the predicted probabilities.

## Repository structure

```
.
├── NLP final.ipynb    # Full pipeline: crawling, preprocessing, model, training, evaluation
├── requirements.txt
└── README.md
```

The notebook was written for Google Colab and reads and writes under `/content/drive/MyDrive/Colab Notebooks/NLP/project/`. Its main sections are:

| Cells | Step |
| --- | --- |
| 0 | Read the raw CSVs, map labels, crawl and resize thumbnails |
| 2–9 | Filter incomplete rows, balance classes, split the data |
| 12 | Model definition, training loop, ensemble evaluation |
| 16–20 | Train the five variants |
| 22–25 | Load the best three and evaluate the ensemble on the test set |
| 30–31 | Predict a single article |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Training used a single A100 (Colab). RoBERTa-large and CLIP are both fine-tuned, so a GPU with 24 GB or more of memory is recommended.

## Limitations

- The dataset files, crawled thumbnails, and trained weights are not included in this repository, so results cannot be reproduced from the repository alone.
- The share of clickbait differs strongly between publishers (most thesun.ie articles are labelled clickbait, while most irishexaminer.com and rte.ie articles are not), so part of what the model learns is publisher writing style.
- The notebook's k-shot cells (27–28) score the fully trained ensemble on 10–40 sampled test articles. They are not few-shot training experiments and are not reported as results.
