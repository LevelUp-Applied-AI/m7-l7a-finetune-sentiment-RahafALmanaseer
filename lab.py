"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os
os.system("pip install --upgrade accelerate")

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# 3-class sentiment label mapping (matches the curated dataset's `label` column)
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.

    The CSV must have at least `text` and `label` columns.
    """
    # read the CSV with pandas
    df = pd.read_csv(data_path)
    # convert with Dataset.from_pandas(df, preserve_index=False)
    ds = Dataset.from_pandas(df, preserve_index=False)
    # split with .train_test_split(test_size=test_size, seed=seed)
    # return the resulting DatasetDict
    return ds.train_test_split(test_size=test_size, seed=seed)


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.
    """
    # define tokenize_fn(batch) calling the passed-in tokenizer with truncation + max_length
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    # apply ds_dict.map(tokenize_fn, batched=True)
    # return the tokenized DatasetDict
    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",  
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
    )

    args.eval_strategy = "epoch"
    args.save_strategy = "epoch"
    
    return args


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.
    """
    # unpack eval_pred to logits, labels
    logits, labels = eval_pred
    # argmax logits over axis 1
    predictions = np.argmax(logits, axis=-1)
    # compute accuracy and macro-F1
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    # return as a dict
    return {"accuracy": acc, "macro_f1": f1}


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.
    """
    # load model with AutoModelForSequenceClassification.from_pretrained(
    # model_name, num_labels=num_labels, id2label=ID2LABEL, label2id=LABEL2ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # build data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # build Trainer with model, args, train/eval datasets, tokenizer, data_collator, compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        processing_class=tokenizer,  

        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # call trainer.train()
    trainer.train()

    # return trainer
    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Returns: {"accuracy": float, "macro_f1": float, "per_class_f1": {label_name: f1, ...}, ...}
    """
    # predict on tokenized_test using trainer.predict
    predictions_output = trainer.predict(tokenized_test)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    # argmax predictions to class indices
    preds = np.argmax(logits, axis=-1)

    # compute accuracy and macro-F1
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    # compute per-class metrics
    f1_per_class = f1_score(labels, preds, average=None)
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

    # build per_class dicts using trainer.model.config.id2label for label names
    id2label = trainer.model.config.id2label
    per_class_f1 = {id2label[i]: float(f1_per_class[i]) for i in range(len(f1_per_class))}
    per_class_precision = {id2label[i]: float(precision_per_class[i]) for i in range(len(precision_per_class))}
    per_class_recall = {id2label[i]: float(recall_per_class[i]) for i in range(len(recall_per_class))}

    # return all metrics
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def main() -> None:
    """Orchestrate the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    ds = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # Save locally (model/ is gitignored)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV
    pred_logits = trainer.predict(tokenized["test"]).predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)
    id2label = trainer.model.config.id2label

    df_out = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": [id2label[i] for i in ds["test"]["label"]],
        "predicted_label": [id2label[i] for i in pred_idx],
        "predicted_probability": [float(pred_probs[i, pred_idx[i]]) for i in range(len(pred_idx))],
    })

    # Add per-class probability columns
    for i, label_name in id2label.items():
        df_out[f"prob_{label_name}"] = pred_probs[:, i]

    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Confusion matrix (for the evaluation report)
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    cm_df = pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values()))
    print(cm_df.to_string())
    cm_df.to_csv("confusion_matrix.csv")

    # Push to Hugging Face Hub.
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/<your-username>/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `python -m huggingface_hub login` and try again.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    main()