# Seq2Seq Transformer for English to German Translation

This project reproduces a complete transformer-based translation pipeline in PyTorch, built from a single notebook. It includes BPE tokenization, a Seq2Seq Transformer architecture, training, and beam search decoding for English to German translation.

Source: *Building a Seq2Seq Transformer Model for Language Translation: A Comprehensive Guide* by Ravjot Singh.

## Dataset
The model is trained on the `Multi30k` dataset (English–German). This dataset contains short, image-caption style sentences, which makes it well-suited for learning basic translation patterns but not for general-domain translation.

### Dataset Biases and Limitations
- **Domain bias:** Sentences are short and descriptive (image captions), so the model may perform poorly on formal, technical, or long-form text.
- **Coverage bias:** The dataset is relatively small compared to modern web-scale corpora, which limits vocabulary coverage and robustness.

## Method
### 1) BPE Tokenization
- SentencePiece BPE tokenizer trained on the Multi30k training split.
- Shared vocabulary for source and target languages.
- Vocabulary size: 10,000
- Character coverage: 1.0
- Byte fallback: enabled

### 2) Model Architecture
A Seq2Seq Transformer (encoder–decoder) implemented with `torch.nn.Transformer`:
- `d_model`: 512
- `nhead`: 8
- Encoder layers: 4
- Decoder layers: 4
- Feedforward dimension: 2048
- Dropout: 0.1
- Weight initialization: Xavier uniform
- Random seed: 0

### 3) Training
- Optimizer: Adam (`lr=3e-4`, `betas=(0.9, 0.98)`, `eps=1e-9`)
- Loss: Cross-Entropy with padding masked
- Batch size: 128
- Epochs: 10

### 4) Decoding
Beam search is used at inference time:
- Beam size `K = 5`
- Length penalty `0.7`
