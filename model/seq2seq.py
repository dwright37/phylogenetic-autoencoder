import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from model.rnn import ConditionalGRUCell, ConditionalGRUState

def gru(input_seq, input_seq_len, gru_dim, batch_size=None, keep_prob=1.0, scope="gru"):
    """
    Defines a GRU and returns the final state

    input_seq (Tensor):         The input sequence to encode
    input_seq_len (Tensor):     The lengths of each sequence in the batch
    gru_dim (int):              The dimensionality of the GRU
    batch_size (Tensor):        The input batch size
    keep_prob (float):          Dropout keep probability
    scope (string):             The name of the variable scope

    Returns: A tensor with the final hidden state
    """
    with tf.variable_scope(scope):
        cell = tf.contrib.rnn.GRUCell(gru_dim)

        if keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=keep_prob)
        
        _, encoder_state  = tf.nn.dynamic_rnn(cell, input_seq, sequence_length=input_seq_len, dtype=tf.float32)
        
    return encoder_state

def lstm(input_seq, input_seq_len, lstm_dim, batch_size=None, keep_prob=1.0, scope="bilstm"):
    """
    Defines a LSTM and returns the final state

    input_seq (Tensor):         The input sequence to encode
    input_seq_len (Tensor):     The lengths of each sequence in the batch
    lstm_dim (int):             The dimensionality of the LSTM
    batch_size (Tensor):        The input batch size
    keep_prob (float):          Dropout keep probability
    scope (string):             The name of the variable scope

    Returns: A tensor with the final hidden state
    """
    with tf.variable_scope(scope):
        cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim)

        if keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, state_keep_prob=keep_prob)
        
        _, encoder_state  = tf.nn.dynamic_rnn(cell, input_seq, sequence_length=input_seq_len, dtype=tf.float32)
        
    return encoder_state

def bilstm(input_seq, input_seq_len, lstm_dim, batch_size=None, keep_prob=1.0):
    """
    Defines a Bidirectional LSTM and returns the final state

    input_seq (Tensor):         The input sequence to encode
    input_seq_len (Tensor):     The lengths of each sequence in the batch
    lstm_dim (int):             The dimensionality of the LSTM
    batch_size (Tensor):        The input batch size
    keep_prob (float):          Dropout keep probability
    scope (string):             The name of the variable scope

    Returns: A tensor with the final hidden state
    """
    with tf.variable_scope("bilstm"):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(lstm_dim)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(lstm_dim)

        if keep_prob < 1.0:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, state_keep_prob=keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, state_keep_prob=keep_prob)
        
        _, ((enc_c_fw, enc_h_fw), (enc_c_bw, enc_h_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                cell_bw, input_seq, sequence_length=input_seq_len, dtype=tf.float32)
            
        encoder_state = (tf.zeros((batch_size, 2*lstm_dim)), tf.concat([enc_h_fw, enc_h_bw], axis=1))

    return encoder_state

def basic_seq2seq(encoder_in, encoder_lens, decoder_in, decoder_lens, lstm_type='fw_lstm',
                        vocab_size=100000, lstm_dim=125, keep_prob=1.0, batch_size=None,
                        l2reg=0.001, inference=False, target_sos_id=0, target_eos_id=0, 
                        max_iterations=20):
    """
    Defines a basic seq2seq model

    encoder_in (Tensor):        The input sequences to the encoder
    encoder_lens (Tensor):      The length of each encoder input sequence
    decoder_in (Tensor):        The input sequences to the decoder
    decoder_lens (Tensor):      The length of each decoder input sequence
    lstm_type (string):         The type of encoder to use. Valid types are "fw_lstm", "bilstm", and "gru"
    vocab_size (int):           The number of tokens in the vocabulary
    lstm_dim (int):             The RNN hidden state size (both encoder and decoder)
    keep_prob (float):          The dropout keep probability
    batch_size (Tensor):        Number of input sequences
    l2reg (float):              L2 regularization parameter
    inference (bool):           True during test time, False for training
    target_sos_id (int):        Id for start of sequence token (inference only)
    target_eos_id (int):        Id for end of sequence token (inference only)
    max_iterations (int):       Maximum number of symbols to generate (inference only)

    Returns: The logits and decoder output for inference
    """

    #Load the embeddings
    with tf.variable_scope("embeddings"):
        #Trying just one-hot at first
        enc_emb = tf.one_hot(encoder_in, vocab_size)
        dec_emb = tf.one_hot(decoder_in, vocab_size)
            
    with tf.name_scope("encoder"):
        #Get the encoder hidden state
        if lstm_type in 'fw_lstm':
            encoder_state = lstm(enc_emb, encoder_lens, lstm_dim, batch_size=batch_size, keep_prob=keep_prob, scope="lstm")
            #encoder_state = tf.contrib.rnn.LSTMStateTuple(c=tf.zeros((batch_size, lstm_dim)), h=encoder_state[1]) 
            encoder_state = ConditionalGRUState(h=tf.zeros((batch_size, lstm_dim)), c=encoder_state[1]) 
            dec_dim = lstm_dim
        elif lstm_type in 'gru':
            encoder_state = gru(enc_emb, encoder_lens, lstm_dim, batch_size=batch_size, keep_prob=keep_prob, scope="gru")
            #encoder_state = tf.contrib.rnn.LSTMStateTuple(c=tf.zeros((batch_size, lstm_dim)), h=encoder_state[1]) 
            encoder_state = ConditionalGRUState(h=tf.zeros((batch_size, lstm_dim)), c=encoder_state) 
            dec_dim = lstm_dim
        else:
            encoder_state = bilstm(enc_emb, encoder_lens, lstm_dim, batch_size=batch_size, keep_prob=keep_prob)
            #encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state[0], h=encoder_state[1])
            encoder_state = ConditionalGRUState(h=tf.zeros((batch_size, 2*lstm_dim)), c=encoder_state[1]) 
            dec_dim = 2*lstm_dim


        with tf.name_scope("logging"):
            tf.summary.histogram("sequence_embedding", encoder_state)

    with tf.name_scope("decoder"):
        #dec_cell = tf.contrib.rnn.BasicLSTMCell(dec_dim)
        dec_cell = ConditionalGRUCell(dec_dim)        

        # Helper
        if inference:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_var, 
                                    tf.fill([batch_size], target_sos_id), target_eos_id)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(dec_emb, decoder_lens)
        
        projection_layer = layers_core.Dense(vocab_size, use_bias=False)
        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            dec_cell, helper, encoder_state,
            output_layer=projection_layer)

        maximum_iterations = None
        if inference:
            maximum_iterations = max_iterations
        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory=True, 
                            maximum_iterations=maximum_iterations, impute_finished=True)
        logits = outputs.rnn_output
        translations = outputs.sample_id

    return logits, translations
