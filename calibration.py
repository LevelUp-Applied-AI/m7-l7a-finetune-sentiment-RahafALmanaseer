import numpy as np
import matplotlib.pyplot as plt
import os

def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    bucket_accuracies = []
    bucket_counts = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i+1]
        mask = (confidences >= bin_lower) & (confidences < bin_upper)
        if i == n_bins - 1: mask = (confidences >= bin_lower) & (confidences <= bin_upper)

        count = np.sum(mask)
        bucket_counts.append(count)
        if count > 0:
            acc = np.mean(predictions[mask] == y_true[mask])
            bucket_accuracies.append(acc)
        else:
            bucket_accuracies.append(0.0)

    return (bin_edges[:-1] + bin_edges[1:]) / 2, np.array(bucket_accuracies), np.array(bucket_counts)

def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if i == n_bins - 1: mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1])
        count = np.sum(mask)
        if count > 0:
            acc = np.mean(predictions[mask] == y_true[mask])
            conf = np.mean(confidences[mask])
            ece += (count / len(y_true)) * np.abs(acc - conf)
    return float(ece)

def plot_reliability(centers, accs, counts, output_path: str):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.bar(centers, accs, width=0.1, alpha=0.8, edgecolor="black", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    if not os.path.exists("probs.npy"):
        print("Error: probs.npy not found. Run manual_eval.py first!")
    else:
        probs = np.load("probs.npy")
        y_true = np.load("y_true.npy")

        ece = expected_calibration_error(probs, y_true)
        print(f"\n--- Calibration Results ---")
        print(f"ECE: {ece:.4f}")

        centers, accs, counts = reliability_diagram(probs, y_true)
        plot_reliability(centers, accs, counts, "figures/reliability-diagram.png")