# Government Complaint Classification — NLP

A machine learning solution for classifying citizen complaints into a combination of **8 complaint categories** and **3 urgency levels**, resulting in **24 possible classes**.

## Problem

Each complaint is assigned a label in the following format:

```text
category|urgency
```

### Categories

- Agriculture & Irrigation
- Education
- Electricity
- Healthcare
- Law & Order
- Roads & Transport
- Water Supply
- Welfare Schemes

### Urgency Levels

- Critical
- High
- Routine

This results in a total of **24 classification labels**.

---

## Dataset

The dataset contains structured and textual information about citizen complaints.

- **Training data:** 4,800 labeled complaints
- **Test data:** 2,000 complaints

Relevant features include:

- `subject`
- `body`
- `channel`
- `district`
- `complaint_history`

The training data contained no missing values.

---

## Approach

Several approaches were evaluated during experimentation, including:

- TF-IDF + LinearSVC
- Dedicated urgency classifiers
- Hierarchical/category-specific classifiers
- Sentence embeddings
- Confidence-based model combinations
- Metadata-enhanced models

The final approach uses **BGE-M3 embeddings** together with structured metadata.

### Final Pipeline

```text
Subject + Body
      │
      ▼
   BGE-M3
      │
      ▼
 1024-dimensional
 text embeddings
      │
      ├─────────────────────┐
      │                     │
      │              Structured Metadata
      │              ├── channel
      │              ├── district
      │              └── complaint_history
      │                     │
      └──────────┬──────────┘
                 ▼
          Combined Features
             1040 dims
                 │
                 ▼
             LinearSVC
                 │
                 ▼
            24-class label
```

### Text Representation

The `subject` and `body` of each complaint are combined and encoded using **BGE-M3**.

The resulting embeddings are normalized and represented as **1024-dimensional vectors**.

### Metadata

The following categorical features are one-hot encoded:

```text
channel
district
complaint_history
```

These produced 16 additional features.

The final feature representation contains:

```text
1024 BGE-M3 features
+
16 metadata features
=
1040 features
```

### Classifier

A `LinearSVC` classifier with:

```text
C = 1.0
```

was trained on the combined representation.

---

## Validation

To avoid relying on a single train/validation split, the final approach was evaluated using **5-fold stratified out-of-fold validation**.

### Results

| Model | 5-Fold OOF Accuracy |
|---|---:|
| TF-IDF baseline | ~94% |
| BGE-M3 | **98.52%** |
| BGE-M3 + Metadata | **98.62%** |

The BGE-M3 + metadata approach improved over BGE-M3 alone by approximately **0.10 percentage points**.

The fold standard deviation was approximately **0.31 percentage points**.

---

## Experiments

The solution was developed through iterative experimentation.

### TF-IDF Baseline

A lexical TF-IDF model provided a strong baseline at approximately **94% accuracy**, but remained substantially behind the embedding-based approach.

### BGE-M3

BGE-M3 produced a major improvement, reaching:

```text
98.52% 5-fold OOF accuracy
```

### Metadata Enhancement

Adding `channel`, `district`, and `complaint_history` provided a further improvement:

```text
98.52% → 98.62%
```

### Dedicated Urgency Model

Since most remaining errors were related to urgency rather than complaint category, a separate urgency classifier was evaluated.

However, its OOF performance was lower than the joint 24-class BGE + metadata model, so it was not used in the final approach.

### Model Ensembling

TF-IDF and BGE-M3 predictions were also compared.

Although TF-IDF contained some complementary information, confidence-gated TF-IDF overrides did not improve the BGE-M3 OOF score.

The simpler **BGE-M3 + metadata** approach was therefore retained.

---

## Error Analysis

The BGE-M3 model showed very strong category-level performance.

On the validation split, category accuracy reached:

```text
100.00%
```

The remaining errors were primarily related to distinguishing between urgency levels within the correct category.

The most common confusion was between:

```text
high ↔ routine
```

This motivated experiments with dedicated urgency models and metadata-based improvements.

The experiments showed that retaining the joint 24-class BGE + metadata classifier was more effective than forcing a separate urgency classification stage.

---

## Final Model

The final model was trained on the complete **4,800-row training dataset** and used to generate predictions for the **2,000-row test dataset**.

The prediction format is:

```text
id,label
```

where `label` follows:

```text
category|urgency
```

Example:

```text
G105529,agriculture_irrigation|high
G105472,electricity|routine
G103379,roads_transport|high
```

---

## Reproducibility

The notebook included in this repository contains the complete modeling workflow, including:

1. Data loading and inspection
2. Text preprocessing
3. BGE-M3 embedding generation
4. Metadata encoding
5. Feature combination
6. Model training
7. Cross-validation experiments
8. Error analysis
9. Final training
10. Test prediction generation
11. Submission validation

The notebook was developed and executed in a **Kaggle environment**.

The BGE-M3 model was added as a Kaggle Model Input rather than stored directly in this repository.

---

## Submission Validation

Before generating the final submission, the predictions were checked for:

- Correct number of rows
- Missing IDs
- Missing labels
- Duplicate IDs
- Invalid labels
- Correct submission columns

All checks passed:

```text
2,000 predictions
No missing IDs
No missing labels
No duplicate IDs
All labels valid
```

---

## Repository Contents

```text
.
├── README.md
└── <notebook>.ipynb
```

The notebook contains the executable implementation along with the saved outputs from the experiments.

---

## Key Takeaway

The experiments showed that **semantic embeddings were considerably more effective than purely lexical features for this classification task**.

BGE-M3 provided a strong representation of the complaint text, while the structured metadata provided an additional measurable improvement.

The final validated approach achieved:

> **98.62% 5-fold out-of-fold accuracy**
