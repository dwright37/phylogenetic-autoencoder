# Phylogenetic autoencoder
Autoencoder which uses a bidirectional GRU to learn useful representations of gene sequences.

## Experiment description
First, each character in a sequence is embedded to a one-hot encoding. With four possible base pairs and an "out of vocab" token, this gives a length 5 input representation. I use an encoder-decoder approach similar to skip-thoughts (Kiros et al., 2015, https://arxiv.org/abs/1506.06726), where the sequence is first embedded to a dense representation using an RNN and subsequently decoded using a conditional RNN. I use a Gated Recurrent Unit (GRU) for the encoding RNN and a conditional GRU for the decoder. Once the entire sequence is passed through the encoder, the final hidden state representation is used as the conditioning vector in the decoder. In this, the decoder output is partially a function of the encoded representation at every time step. The decoder hidden state is initialized to all zeros. It is trained to predict the correct character at each step of the original input sequence. The model is trained using cross-entropy loss between the ground truth characters and the predicted characters. I use the Adam optimizer with an initial learning rate of 0.001 (all other parameters tensorflow defaults) and L2 regularization (0.01). I use 100 character green genes sequences as the training set, with a testing set of 8012 sequences. The model converges after ~13 epochs through the training set.

## Module contents
- utils/: Text processing utilities
- data/: Contains the training/test data and vocabulary
- model/: Python modules defining the tensorflow models
- preprocess\_corpus.ipynb: Python notebook to preprocess the text corpora and generate data which is fed to the models
- train\_seq2seq.ipynb: Training notebook
- visualize\_embedding\_space.ipynb: Visualization and embedding generation notebook

## NOTE
Make sure to run the cells in preprocess\_corpus.ipynb to generate the data which is fed into the models.

## Visualization

