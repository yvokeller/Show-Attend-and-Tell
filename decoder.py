import torch
import torch.nn as nn
from attention import Attention

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, tf=False, ado=False, bert=False, attention=False):
        super(Decoder, self).__init__()
        self.use_tf = tf
        self.use_advanced_deep_output = ado
        self.use_bert = bert
        self.use_attention = attention

        # Initializing parameters
        self.encoder_dim = encoder_dim

        # Embeddings
        if bert == True:
            from transformers import BertModel, BertTokenizer
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.vocabulary_size = self.bert_model.config.vocab_size
            self.embedding_size = self.bert_model.config.hidden_size # 768

            # Embedding layer using BERT's embeddings
            self.embedding = self.bert_model.get_input_embeddings()

            # Freeze the BERT embeddings
            for param in self.embedding.parameters():
                param.requires_grad = False

            # Delete the BERT model to save memory (and checkpoint size)
            del self.bert_model
        else:
            self.vocabulary_size = vocabulary_size
            self.embedding_size = 512
            self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)  # Embedding layer for input words

        # Initial LSTM cell state generators
        self.init_h = nn.Linear(encoder_dim, self.embedding_size)  # For hidden state
        self.init_c = nn.Linear(encoder_dim, self.embedding_size)  # For cell state
        self.tanh = nn.Tanh()

        # Attention mechanism related layers
        self.f_beta = nn.Linear(self.embedding_size, encoder_dim)  # Gating scalar in attention mechanism
        self.sigmoid = nn.Sigmoid()

        # Attention and LSTM components
        self.attention = Attention(encoder_dim, self.embedding_size)  # Attention network
        self.lstm = nn.LSTMCell(self.embedding_size + encoder_dim, self.embedding_size)  # LSTM cell

        # Deep output layers
        if self.use_advanced_deep_output:
            # Advanced DO: Layers for transforming LSTM state, context vector and embedding for DO-RNN
            hidden_dim, intermediate_dim = self.embedding_size, self.embedding_size
            self.f_h = nn.Linear(hidden_dim, intermediate_dim)  # Transforms LSTM hidden state
            self.f_z = nn.Linear(encoder_dim, intermediate_dim)  # Transforms context vector
            self.f_out = nn.Linear(intermediate_dim, self.vocabulary_size)  # Transforms the combined vector (sum of embedding, LSTM state, and context vector) to vocabulary
            self.relu = nn.ReLU()  # Activation function
            self.dropout = nn.Dropout()
        
        # Simple DO: Layer for transforming LSTM state to vocabulary
        self.deep_output = nn.Linear(self.embedding_size, self.vocabulary_size)  # Maps LSTM outputs to vocabulary
        self.dropout = nn.Dropout()

    def forward(self, img_features, captions):
        # Forward pass of the decoder
        batch_size = img_features.size(0)

        # Initialize LSTM state
        h, c = self.get_init_lstm_state(img_features)

        # Teacher forcing setup
        max_timespan = max([len(caption) for caption in captions]) - 1

        if self.use_bert:
            start_token = torch.full((batch_size, 1), self.tokenizer.cls_token_id).long().to(mps_device)
        else:
            start_token = torch.zeros(batch_size, 1).long().to(mps_device)
        
        # Convert caption tokens to their embeddings
        if self.use_tf:
            # current_embedding = self.embedding(captions) if self.training else self.embedding(prev_words) # TODO: I think this else case never happens
            caption_embedding = self.embedding(captions)
        else:
            previous_predicted_token_embedding = self.embedding(start_token)

        # Preparing to store predictions and attention weights
        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(mps_device) # [BATCH_SIZE, TIME_STEPS, VOC_SIZE] = one-hot encoded prediction of token for each time step
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(mps_device) # [BATCH_SIZE, TIME_STEPS, NUM_SPATIAL_FEATURES] = attention weight of each feature map for each time step

        # Generating captions
        for t in range(max_timespan):
            if self.use_attention:
                context, alpha = self.attention(img_features, h)  # Compute context vector via attention
                gate = self.sigmoid(self.f_beta(h))  # Gating scalar for context
                gated_context = gate * context  # Apply gate to context
            else:
                # If not using attention, treat all parts of the image equally
                alpha = torch.full((batch_size, img_features.size(1)), 1.0 / img_features.size(1), device=mps_device)  # Uniform attention
                context = img_features.mean(dim=1)  # Simply take the mean of the image features
                gated_context = context  # No gating applied

            # Prepare LSTM input
            if self.use_tf:
                lstm_input = torch.cat((caption_embedding[:, t], gated_context), dim=1)  # current embedding + context vector as input vector
            else:
                previous_predicted_token_embedding = previous_predicted_token_embedding.squeeze(1) if previous_predicted_token_embedding.dim() == 3 else previous_predicted_token_embedding # TODO: What is the optional squeeze for?
                lstm_input = torch.cat((previous_predicted_token_embedding, gated_context), dim=1)

            # LSTM forward pass
            h, c = self.lstm(lstm_input, (h, c))

            # Generate word prediction
            if self.use_advanced_deep_output:
                # NOTE: could explore alternative positions for dropout
                if self.use_tf:
                    output = self.advanced_deep_output(self.dropout(h), context, caption_embedding[:, t])
                else:
                    output = self.advanced_deep_output(self.dropout(h), context, previous_predicted_token_embedding)
            else:
                output = self.deep_output(self.dropout(h))

            preds[:, t] = output  # Store predictions
            alphas[:, t] = alpha  # Store attention weights

            # Prepare next input word
            if not self.use_tf: # NOTE: my understanding was this needs to be done for training with tf too? -> WRONG, because for tf we already have the whole caption and don't want to overwrite with the predicted word
                predicted_token_idxs = output.max(1)[1].reshape(batch_size, 1) # output.max(1)[1] = extract the index: [1] of the token with the highest probability: max(1)
                previous_predicted_token_embedding = self.embedding(predicted_token_idxs)

        return preds, alphas

    def get_init_lstm_state(self, img_features):
        # Initializing LSTM state based on image features
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)  # Cell state
        c = self.tanh(c)

        h = self.init_h(avg_features)  # Hidden state
        h = self.tanh(h)

        return h, c
    
    def advanced_deep_output(self, h, context, current_embedding):
        # Combine the LSTM state and context vector
        h_transformed = self.relu(self.f_h(h))
        z_transformed = self.relu(self.f_z(context))

        # Sum the transformed vectors with the embedding
        combined = h_transformed + z_transformed + current_embedding

        # Transform the combined vector & compute the output word probability
        return self.relu(self.f_out(combined))

    def caption(self, img_features, beam_size):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        if self.use_bert == True:
            prev_words = torch.full((beam_size, 1), self.tokenizer.cls_token_id).long()
        else:
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

            if self.use_attention:
                context, alpha = self.attention(img_features, h)  # Compute context vector via attention
                gate = self.sigmoid(self.f_beta(h))  # Gating scalar for context
                gated_context = gate * context  # Apply gate to context
            else:
                # If not using attention, treat all parts of the image equally
                batch_size = img_features.shape[0]
                alpha = torch.full((batch_size, img_features.size(1)), 1.0 / img_features.size(1))  # Uniform attention
                context = img_features.mean(dim=1)  # Simply take the mean of the image features
                gated_context = context  # No gating applied

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            if self.use_advanced_deep_output:
                current_embedding = self.embedding(prev_words).squeeze(1)
                output = self.advanced_deep_output(h, context, current_embedding)
            else:
                output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True) # beamsize, dim, largest, sorted
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words // output.size(1) # calculates the indices of the previous words in the sequences
            next_word_idxs = top_words % output.size(1) # calculates the indices of the next words to add to each sequence.

            prev_word_idxs = prev_word_idxs.long()
            next_word_idxs = next_word_idxs.long()

            # DEBUGGING: print next_word_idxs with their likelihood
            # for i, next_word_idx in enumerate(next_word_idxs):
            #    print(f'Next word {i}: {self.tokenizer.decode(next_word_idx)} with likelihood {top_preds[i].item()}')

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            # creates a list containing the indices of all sequences that are not yet complete
            if self.use_bert:
                # Quickfix for training BERT with SEP token at the end of sequence, PAD inbetween...
                incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1 and next_word != 0]
            else:
                # TODO: normal behaviour, revert to this later
                incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1 and next_word != 102] # 1: <eos>, 102: [SEP]
            
            # contains indices of sequences that have reached a conclusion
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break

            # update tensor information to keep the information for the incomplete sequences
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1) # extract the top predictions for the incomplete sequences
            prev_words = next_word_idxs[incomplete].unsqueeze(1) # next_words of incomplete sequences become the prev_words for the next iteration

            if step > 50:
                break
            step += 1

        if len(completed_sentences_preds) == 0:
            print('No completed sentences found')
            return [0], alpha

        # Print all completed sentences if BERT is used
        # if self.use_bert:
        #    for i, sentence in enumerate(completed_sentences):
        #        print(f'Sentence {i}: {self.tokenizer.decode(sentence, skip_special_tokens=True)}')

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
  
        return sentence, alpha
