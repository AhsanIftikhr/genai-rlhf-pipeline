# GenAI RLHF Pipeline: SFT, Reward Modeling, PPO & DPO

An implementation of a Generative AI alignment pipeline using **SmolLM2-135M**, covering supervised fine-tuning, reward modeling, and preference optimization techniques.

## Project Overview

This project explores how a pretrained language model can be fine-tuned and aligned with human preferences using different training approaches.

The implementation uses Google Colab with GPU acceleration and Hugging Face libraries.

## Objectives

- Understand and implement Supervised Fine-Tuning (SFT).
- Train a Reward Model using preference data.
- Implement Proximal Policy Optimization (PPO).
- Implement Direct Preference Optimization (DPO).
- Evaluate and compare the resulting models.

## Base Model

- **Model:** SmolLM2-135M
- **Framework:** PyTorch
- **Platform:** Google Colab
- **Libraries:** Transformers, Datasets, Hugging Face

## Project Workflow

1. Environment setup and configuration
2. Tokenizer loading
3. Base model loading
4. Dataset preparation
5. Supervised Fine-Tuning (SFT)
6. Reward Model training
7. PPO optimization
8. DPO optimization
9. Model evaluation and comparison

## Datasets

Two datasets are used in the implementation:

### SFT Dataset
- Training: 116,722 examples
- Testing: 6,553 examples
- Validation: 6,447 examples

### Preference Dataset
- Training: 92,534 examples
- Testing: 83,629 examples
- Validation splits: 33,082 and 50,715 examples

## Training Results

### Supervised Fine-Tuning

| Metric | Result |
|---|---:|
| Optimizer steps | 1,250 |
| Final training loss | 2.3345 |
| Final validation loss | 2.3534 |
| Training time | 71.56 minutes |

### Reward Model

| Metric | Result |
|---|---:|
| Training steps | 2,500 |
| Final validation loss | 0.6648 |
| Final pairwise accuracy | 62.0% |
| Training time | 137.73 minutes |

### PPO and DPO

Implementation and evaluation results will be documented after completing both optimization stages.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Google Colab
- CUDA / GPU

## Repository Structure

```text
genai-rlhf-pipeline/
│
├── notebooks/
│   └── Gen_AI_Assignment.ipynb
│
├── results/
│   ├── sft_loss.png
│   ├── reward_loss.png
│   ├── reward_accuracy.png
│   └── reward_final_results.json
│
├── README.md
└── .gitignore
```

## How to Run

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Install the required Python dependencies.
4. Load the datasets and pretrained model.
5. Execute the notebook cells in sequence.
6. Train and evaluate the SFT, Reward, PPO, and DPO stages.

## Note

Model checkpoints and large datasets are not included in this repository. They can be downloaded or generated separately.

## Author

**Ahsan Iftikhar**

AI Graduate | Generative AI & Machine Learning


---

*This project is developed as part of a Generative AI assignment to explore RLHF and preference optimization techniques.*
