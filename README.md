## Overview

This repository contains code for WIPER (WIPER: Defense against Backdoor Attacks via Identifying and Purifying Bad Neurons).

1.  **Training a backdoor model:** An innocent-looking model is trained on a dataset with a small percentage of poisoned data. This poisoned data contains a hidden trigger that, when present, forces the model to misclassify inputs to a specific target class.

2.  **Defending against the attack:** WIPER is implemented to "purify" the backdoored model by modifying the model's weights and biases to remove the malicious behavior without significantly degrading its performance on clean data.

## How to Run

### 1\. Train the Backdoored Model

First, you need to train a model with the backdoor. The `training_backdoor_model.py` script handles this process. By default, it will train a `ResNet18` model on the `CIFAR-10` dataset with a `5%` poisoning rate and save the poisoned model to the `saved_models` directory. You can change hyperparameters to specify different backdoor attacks.

```bash
python purifying_backdoor/training_backdoor_model.py
```

### 2\. Apply WIPER

Once you have a backdoored model, you can run the defense script to "purify" it. The `defense.py` script loads the saved poisoned model and applies our defense WIPER.

```bash
python purifying_backdoor/defense.py
```
