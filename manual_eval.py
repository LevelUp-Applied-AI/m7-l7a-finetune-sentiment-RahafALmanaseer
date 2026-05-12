import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
# Import helpers from your lab.py
from lab import prepare_dataset, tokenize_dataset, get_data_path

def manual_predict(model, tokenizer, dataset, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Manual PyTorch inference loop."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)

def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """Compute metrics using only numpy."""
    labels = np.unique(y_true)
    accuracy = np.mean(y_true == y_pred)
    per_class = {}

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class[int(label)] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    macro_f1 = np.mean([stats["f1"] for stats in per_class.values()])
    return {"accuracy": float(accuracy), "macro_f1": float(macro_f1), "per_class": per_class}

if __name__ == "__main__":
    # Path to your local model from Lab 7A
    model_path = "./model"

    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' not found!")
    else:
        # 1. Load data
        data_path = get_data_path()
        ds = prepare_dataset(data_path)

        # 2. Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # 3. Tokenize and format
        tokenized_ds = tokenize_dataset(ds, tokenizer)
        test_split = tokenized_ds["test"]
        test_split.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        # 4. Predict
        print("Running manual inference...")
        preds, probs = manual_predict(model, tokenizer, test_split)
        y_true = np.array(test_split["label"])

        # 5. Report
        report = compute_classification_report_from_arrays(y_true, preds)
        print("\n--- Manual Classification Report ---")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro-F1: {report['macro_f1']:.4f}")

        # Save results for calibration script
        np.save("probs.npy", probs)
        np.save("y_true.npy", y_true)
        print("\nResults saved to probs.npy and y_true.npy")