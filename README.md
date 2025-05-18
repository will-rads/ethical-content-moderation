# ethical-content-moderation
Fine-Tuning DistilBERT for Ethical Content Moderation

---
library_name: transformers
license: mit
base_model: distilbert-base-uncased
tags:
- generated_from_keras_callback
model-index:
- name: distilbert-hatespeech-classifier
---

<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

HF Model Card: https://huggingface.co/will-rads/distilbert-hatespeech-classifier

## Model description

This model fine-tunes distilbert-base-uncased on the Davidson et al. (2017) hate speech and offensive language dataset loaded from HuggingFace. The classifier predicts whether a tweet is:

- (a) hate speech
- (b) offensive but not hate
- (c) neither

Using a frozen DistilBERT base and a custom dense head.

The architecture consists of three dense layers (256 → 128 → 32, LeakyReLU and Swish activations), with dropout and batch normalization to improve generalization.


## Intended uses & limitations

Intended uses

- As a starting point for transfer learning in NLP and AI ethics projects

- Academic research on hate speech and offensive language detection

- As a fast, lightweight screening tool for moderating user-generated content (e.g., tweets, comments, reviews)

Limitations
Not suitable for real-time production use without further robustness testing

Trained on English Twitter data (2017) — performance on other domains or languages may be poor

Does not guarantee removal of all forms of bias or unfairness; see Fairness & Bias section

## Training and evaluation data

Dataset:
Davidson et al., 2017 (24K+ English tweets, labeled as hate, offensive, or neither)

Class distribution: Imbalanced (majority: “offensive”; minority: “hate”)

Split: 80% training, 20% validation (stratified)


## Training procedure

Frozen base: DistilBERT transformer weights frozen; only dense classifier head is trained.

Loss: Sparse categorical crossentropy

Optimizer: Adam (learning rate = 3e-5)

Batch size: 16

Class weighting: Used to compensate for class imbalance (higher weight for “hate”)

Early stopping: Custom callback at val_accuracy ≥ 0.92 

Hardware: Google Colab (Tesla T4 GPU)

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'Adam', 'weight_decay': None, 'clipnorm': None, 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99, 'ema_overwrite_frequency': None, 'jit_compile': True, 'is_legacy_optimizer': False, 'learning_rate': np.float32(3e-05), 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}
- training_precision: float32

### Training results

| Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | Epoch |
|:----------:|:--------------:|:---------------:|:-------------------:|:-----:|
| 1.4634     | 0.4236         | 0.9268          | 0.6454              | 1     |
| 1.1659     | 0.5067         | 0.9578          | 0.6480              | 2     |
| 1.0965     | 0.5388         | 0.8224          | 0.7043              | 3     |
| 1.0026     | 0.5667         | 0.8131          | 0.7051              | 4     |
| 0.9948     | 0.5817         | 0.8264          | 0.6940              | 5     |
| 0.9631     | 0.5921         | 0.7893          | 0.7111              | 6     |
| 0.9431     | 0.6009         | 0.7725          | 0.7252              | 7     |
| 0.9019     | 0.6197         | 0.8177          | 0.7049              | 8     |
| 0.8790     | 0.6247         | 0.7408          | 0.7351              | 9     |
| 0.8578     | 0.6309         | 0.7786          | 0.7176              | 10    |
| 0.8275     | 0.6455         | 0.7387          | 0.7331              | 11    |
| 0.8530     | 0.6411         | 0.7253          | 0.7273              | 12    |
| 0.8197     | 0.6506         | 0.7430          | 0.7293              | 13    |
| 0.8145     | 0.6549         | 0.7535          | 0.7162              | 14    |
| 0.8081     | 0.6631         | 0.7207          | 0.7402              | 15    |

### Best validation accuracy:
0.7402 at epoch 15

### Environmental Impact
Training emissions:
Estimated at 0.0273 kg CO₂ (CodeCarbon, Colab T4 GPU)

### Fairness & Bias

Bias/fairness audit:
The model was evaluated on synthetic gender pronoun tests and showed relatively balanced outputs, but biases may remain due to dataset limitations. 
See Appendix B of the project report for details.

### If you use this model, please cite:

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. ICWSM 2017.

 William Radiyeh. DistilBERT Hate Speech Classifier (2025). https://huggingface.co/will-rads/distilbert-hatespeech-classifier


### Framework versions

- Transformers 4.51.3
- TensorFlow 2.18.0
- Datasets 3.6.0
- Tokenizers 0.21.1
