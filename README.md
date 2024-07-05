# Neural Machine Translation with PyTorch

This repository contains code for training and evaluating a Neural Machine Translation (NMT) model using PyTorch. The model translates German sentences to English using sequence-to-sequence architecture with LSTM cells.

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/BytefulRashi/NeuralTranslate/.git
   cd your-repo
   ```

2. **Install dependencies**

   Ensure you have Python 3.7+ installed. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download SpaCy language models**

   Download the required SpaCy language models for German and English:

   ```bash
   python -m spacy download en
   python -m spacy download de
   ```

4. **Run the notebook**

   Open and run the `MachineTranslation.ipynb` notebook using Jupyter or Google Colab. Follow the instructions within the notebook to train the model, evaluate its performance using BLEU score, and perform translations.

## Dataset

- Multi30k dataset is used, which contains parallel English-German sentence pairs.

## Model Architecture

- **Encoder**: LSTM-based encoder that processes German sentences and generates hidden states and cell states.
- **Decoder**: LSTM-based decoder that takes encoder outputs and generates English translations.
- **Seq2Seq**: Sequence-to-sequence model that integrates the encoder and decoder.

## Training

- The model is trained using Adam optimizer with CrossEntropyLoss. Teacher forcing is used during training.

## Evaluation

- BLEU score is used to evaluate the model's translation quality on the test set.

## Results

- After training, the model achieves a BLEU score of 18.98 on the test set.

## Future Improvements

- Experiment with different architectures (e.g., Transformer) for better performance.
- Incorporate attention mechanisms to improve translation quality.
- Expand the dataset for improved model generalization.

## Acknowledgments

- This project is inspired by [Tutorial Link](https://www.youtube.com/watch?v=EoGUlvhRYpk&list=RDCMUCkzW5JSFwvKRjXABI-UTAkQ).
- Special thanks to the contributors of the Multi30k dataset.
