# Module 7 Week A — Lab Evaluation Report

## Dataset
The model was fine-tuned using the **AARSynth app reviews** dataset (Sentences-50Agree). This dataset contains mobile app reviews labeled into three categories: **Positive, Neutral, and Negative**. The data was split into 80% for training and 20% for testing to ensure a robust evaluation of the classifier’s performance.

## Model and Hyperparameters
- **Backbone:** `distilbert-base-uncased`
- **Number of labels:** 3
- **Learning rate:** `5e-5`
- **Epochs:** `2`
- **Batch size:** `8`
- **Max length:** `128`
- **Seed:** `42`
- **Training time:** Approximately **21 minutes** on a local CPU environment.

## Metrics on the test split

### Aggregate Metrics:

| Metric | Value |
|---|---|
| Accuracy | 0.6388 |
| Macro-F1 | 0.6380 |

### Per class (from `metrics.json`):

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Positive | 0.6934 | 0.7230 | 0.6660 |
| Neutral  | 0.4985 | 0.4712 | 0.5292 |
| Negative | 0.7223 | 0.7335 | 0.7114 |

## Confusion Matrix

| | Pred Negative | Pred Neutral | Pred Positive |
|---|---|---|---|
| **True Negative** | 355 | 128 | 16 |
| **True Neutral** | 98 | 245 | 120 |
| **True Positive** | 31 | 147 | 355 |

## Three Qualitative Error Examples

### 1. Mixed Sentiment Confusion (Positive -> Neutral)
- **Sentence:** `"good, but slow workflow."`
- **Gold Label:** `positive`
- **Predicted Label:** `neutral`
- **Gold Class Probability:** `0.302`
- **Reasoning:** The sentence contains both a positive keyword ("good") and a negative observation ("slow"). The model likely struggled with this trade-off and opted for a neutral classification as a "middle ground" rather than identifying the overall positive intent.

### 2. Keyword Bias (Neutral -> Positive)
- **Sentence:** `"nice app to use with friends"`
- **Gold Label:** `neutral`
- **Predicted Label:** `positive`
- **Gold Class Probability:** `0.101`
- **Reasoning:** The presence of the adjective "nice" strongly biased the model toward a positive prediction. While the labeler viewed this as a neutral/descriptive statement, the model’s pre-training likely associates "nice" almost exclusively with positive sentiment.

### 3. Short Sentence Ambiguity (Negative -> Neutral)
- **Sentence:** `"a lot of a bug here"`
- **Gold Label:** `negative`
- **Predicted Label:** `neutral`
- **Gold Class Probability:** `0.460`
- **Reasoning:** The model was very close to the correct answer (46% negative vs 46.9% neutral). The failure to tip into the negative category might be due to the short length of the sentence or the word "bug" appearing in more objective contexts during training.

## Confusion Matrix Interpretation
The model performs best on the **Negative** class (F1: 0.72), effectively identifying complaints. However, the **Neutral** class is the hardest to identify (F1: 0.50). The most significant confusion occurs between **Neutral** and **Positive** labels (147 positive reviews were predicted as neutral). This suggests that the boundary between a "functional" review and a "positive" review is quite thin for this model.

## Hugging Face Hub model URL
https://huggingface.co/Rahaf2002/m7-app-review-sentiment
