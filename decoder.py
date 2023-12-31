import torch
import torch.nn as nn
from attention import Attention

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, tf=False, ado=False):
        super(Decoder, self).__init__()
        self.use_tf = tf
        self.use_advanced_deep_output = ado

        # Initializing parameters
        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        # Initial LSTM cell state generators
        self.init_h = nn.Linear(encoder_dim, 512)  # For hidden state
        self.init_c = nn.Linear(encoder_dim, 512)  # For cell state
        self.tanh = nn.Tanh()

        # Attention mechanism related layers
        self.f_beta = nn.Linear(512, encoder_dim)  # Gating scalar in attention mechanism
        self.sigmoid = nn.Sigmoid()

        # Attention and LSTM components
        self.attention = Attention(encoder_dim)  # Attention network
        self.embedding = nn.Embedding(vocabulary_size, 512)  # Embedding layer for input words
        self.lstm = nn.LSTMCell(512 + encoder_dim, 512)  # LSTM cell

        # Simple DO: Layer for transforming LSTM state to vocabulary
        self.deep_output = nn.Linear(512, vocabulary_size)  # Maps LSTM outputs to vocabulary
        self.dropout = nn.Dropout()

        # Advanced DO: Layers for transforming LSTM state, context vector and embedding for DO-RNN
        hidden_dim, intermediate_dim = 512, 512
        self.f_h = nn.Linear(hidden_dim, intermediate_dim)  # Transforms LSTM hidden state
        self.f_z = nn.Linear(encoder_dim, intermediate_dim)  # Transforms context vector
        self.f_out = nn.Linear(intermediate_dim, vocabulary_size)  # Transforms the combined vector (sum of embedding, LSTM state, and context vector) to vocabulary
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout()

    def forward(self, img_features, captions):
        # Forward pass of the decoder
        batch_size = img_features.size(0)

        # Initialize LSTM state
        h, c = self.get_init_lstm_state(img_features)

        # Teacher forcing setup
        max_timespan = max([len(caption) for caption in captions]) - 1
        prev_words = torch.zeros(batch_size, 1).long().to(mps_device)
        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)

        # Preparing to store predictions and attention weights
        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(mps_device)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(mps_device)

        # Generating captions
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)  # Compute context vector via attention
            gate = self.sigmoid(self.f_beta(h))  # Gating scalar for context
            gated_context = gate * context  # Apply gate to context

            # Prepare LSTM input
            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            # LSTM forward pass
            h, c = self.lstm(lstm_input, (h, c))

            # Generate word prediction
            if self.use_advanced_deep_output:
                # TODO: explore alternative positions for dropout
                output = self.advanced_deep_output(self.dropout(h), context, captions, embedding, t)
            else:
                output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha  # Store attention weights

            # Prepare next input word
            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        # Initializing LSTM state based on image features
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)  # Cell state
        c = self.tanh(c)

        h = self.init_h(avg_features)  # Hidden state
        h = self.tanh(h)

        return h, c
    
    def advanced_deep_output(self, h, context, captions, embedding, t):
        # Combine the LSTM state and context vector
        h_transformed = self.relu(self.f_h(h))
        z_transformed = self.relu(self.f_z(context))

        # Sum the transformed vectors with the embedding
        # TODO: check if embedding is correct for non-training mode
        combined = h_transformed + z_transformed + self.embedding(captions[:, t].long() if self.training else embedding[:, t].long())

        # Transform the combined vector & compute the output word probability
        return self.relu(self.f_out(combined))

    def caption(self, img_features, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            prev_word_idxs = prev_word_idxs.long()
            next_word_idxs = next_word_idxs.long()

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        if len(completed_sentences_preds) == 0:
            print('No completed sentences found')
        
        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha
