# Calibration Analysis

## Reliability diagram interpretation
The reliability diagram saved in (`figures/reliability-diagram.png`) shows the model's performance across different confidence bins. 
* **Interpretation:** The model is consistently **Over-confident**. All bars for confidence levels above 0.4 fall below the dashed "Perfect Calibration" line. 
* **Specific values:** For instance, in the highest confidence bin (**0.9 - 1.0**), the model's actual accuracy is approximately **0.87**, meaning it is about 10-13% less accurate than its confidence score suggests. This gap is visible across all bins from 0.5 to 1.0.

## Expected Calibration Error
The calculated Expected Calibration Error (ECE) is: **0.1040**.
* **Meaning:** On average, there is a **10.4%** mismatch between the model's predicted probabilities and its true accuracy.
* **Trustworthiness for Production:** An ECE of ~10% is moderate. While the model is generally useful for sentiment classification (Accuracy: 63.88%), its probability scores cannot be taken at face value for high-stakes decision-making. A "90% confidence" score only translates to roughly 87% real-world reliability, which must be accounted for in downstream logic.

## A specific calibration pattern
The model displays a classic **Over-confidence on Majority Predictions**. 
* **Reasoning:** This often happens with fine-tuned Transformers like DistilBERT because the `Cross-Entropy Loss` objective penalizes incorrect labels heavily, driving the model to push its Logits to extremes to reach near-1.0 probabilities. Even when the model is wrong or the text is ambiguous, it has learned to "commit" strongly to a class to satisfy the training objective, leading to the observed calibration gap.

## A proposed engineering action
To improve the model's reliability for production, I propose the following:
1. **Temperature Scaling:** This is a post-processing technique where we divide the Logits by a learned parameter $T$ (Temperature) before the Softmax. This would "soften" the distribution and pull the bars closer to the diagonal line without changing the model's accuracy.
2. **Abstention Logic:** Given that accuracy drops below 50% for confidence scores near 0.4-0.5, we could implement a rule to "abstain" or flag for human review any prediction where the confidence is below 0.6.