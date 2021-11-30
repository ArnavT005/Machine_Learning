import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as RNN

# image encoder class (no attention module)
class Encoder(nn.Module):
    # constructor for class Encoder
    # embed_size: size of embedding (latent vector size)
    # batch_size: batch size during mini-batch optimization
    def __init__(self, embed_size, batch_size):
        super(Encoder, self).__init__()
        # set the embedding size (latent vector size fed into decoder)
        self.embed_size = embed_size
        # set batch size (for mini-batch optimizations)
        self.batch_size = batch_size
        # describe the CNN architecture
        # Key Highlights:
        # 1) Used 8 convolutional layers and a maximum of 512 kernel maps per layer
        # 2) Used ReLU as the activation function to avoid vanishing gradient and have faster convergence
        # 3) Used Max-Pooling after every 1-3 convolutional layers for scalability, robustness and to reduce noise (overfitting)
        # 4) Used batch normalization to standardize layer input (faster convergence) and provide some regularization (avoid overfitting)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        # describe the Linear layer
        # Key Highlights:
        # 1) Maps the 3D feature map obtained from CNN into a 1D latent vector
        # 2) Used Tanh as the activation function (since LSTMCell also uses it)
        # 3) Used batch normalization to standardize input (faster convergence) and avoid overfitting 
        self.linear = nn.Sequential(nn.Linear(512*10*10, self.embed_size), nn.BatchNorm1d(self.embed_size), nn.Tanh())    
    # forward function for class Encoder
    # image: batch of image tensor (size = (batch_size, 3, 256, 256))
    def forward(self, image):
        # get the 3D feature map from image
        feature_map = self.cnn(image)                       # size = (batch_size, 512, 10, 10)
        # flatten the feature map
        feature_map = feature_map.view(self.batch_size, -1) # size = (batch_size, 51200)
        # return latent vector
        return self.linear(feature_map)                     # size = (batch_size, embed_size)

# caption decoder class (no attention module)
class Decoder(nn.Module):
    # constructor for class Decoder
    # embed_size; size of embeddings (LSTM input)
    # hidden_size: size of hidden state (LSTM)
    # vocab_size: size of vocabulary dictionary
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        # set the embedding size (this will be the size of the input vector at every time step)
        self.embed_size = embed_size
        # set the hidden size (this will be the size of the hidden state vector (LSTM))
        self.hidden_size = hidden_size
        # set the vocabulary size (this is the size of the vocabulary dictionary)
        self.vocab_size = vocab_size
        # define embedding layer
        # Key Highlights:
        # 1) Generates embeddings with random weights for every word in the vocabulary
        # 2) Treates index 0 as padding ('[PAD]' token)
        # 3) Aim is to learn these weights
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # define LSTM architecture
        # Key Highlights:
        # 1) Takes input vector of size embed_size and produces hidden state vectors of size hidden_size
        # 2) Models the RNN by taking inputs over time and sharing weights
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # define Linear layer
        # Key Highlights:
        # 1) Used to convert the hidden state output of LSTM at each time step into a vocabulary vector
        # 2) This vocabulary vector is thereafter used for loss computation and inference
        # 3) Used dropout layer with probability = 0.5 to avoid overfitting
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    # forward function of class Decoder
    # latent: latent vector received from Encoder (size = (batch_size, embed_size))
    # captions: batch of image captions for current batch (size = (batch_size, max_length))
    # lengths: batch of caption lengths (excluding padding) (size = (batch_size,))
    def forward(self, latent, captions, lengths):
        #self.lstm.flatten_parameters()
        # generate caption embeddings
        embeddings = self.embed(captions)                                               # size = (batch_size, max_length, embed_size)                         
        # concatenate latent vector with word embeddings (to feed into LSTM)
        lstm_input = torch.cat((latent.unsqueeze(1), embeddings), dim=1)                # size = (batch_size, max_length + 1, embed_size)
        # pack the input sequence to remove paddings ('[PAD]' token)
        # lengths vector stores the actual lengths of individual captions, so we also remove the '[END]' token
        packed_seq = RNN.pack_padded_sequence(lstm_input, lengths, batch_first=True)    # size = (sum(lengths), embed_size)
        # pass the input through LSTM
        # short_term represents the short-term LSTM hidden_state (packed currently)
        short_term, _ = self.lstm(packed_seq)                                           # size = (sum(lengths), hidden_size)
        # generate final output after unpacking short_term and applying dropout followed by linear layer
        return self.linear(self.dropout(short_term[0]))                                 # size = (sum(lengths), vocab_size)

# image captioning class (no attention module)
class ImageCaptionsNet(nn.Module):
    # constructor for class ImageCaptionsNet
    # embed_size; size of embeddings (LSTM input)
    # hidden_size: size of hidden state (LSTM)
    # vocab_size: size of vocabulary dictionary
    # batch_size: batch size during mini-batch optimization
    def __init__(self, embed_size, hidden_size, vocab_size, batch_size):
        super(ImageCaptionsNet, self).__init__()
        # set embeddding size
        self.embed_size = embed_size
        # set hidden size
        self.hidden_size = hidden_size
        # set vocabulary size
        self.vocab_size = vocab_size
        # set batch size
        self.batch_size = batch_size
        # instantiate encoder object
        self.encoder = Encoder(embed_size, batch_size)
        # instantiate decoder object
        self.decoder = Decoder(embed_size, hidden_size, vocab_size)
    # forward function of class ImageCaptionsNet
    # image: batch of image tensors (size = (batch_size, 3, 256, 256))
    # captions: batch of image captions (size = (batch_size, max_length))
    # lengths: batch of actual captions length (excluding padding)
    def forward(self, image, captions, lengths):
        # get latent vector from encoder
        latent = self.encoder(image)
        # return the decoder output (used for loss calculation and inference)
        return self.decoder(latent, captions, lengths)
    # caption generator for class ImageCaptionsNet (uses beam search)
    # image: input image (size = (1, 3, 256, 256))
    # vocabulary: reverse vocabulary dictionary (idx2word)
    # beam_size: beam size to be used during beam search (default 5)
    # max_length: maximum length of caption generated (default 10)
    def caption_generator(self, image, vocabulary, beam_size=5, max_length=10):
        # do not use memory for gradients (inference time)
        with torch.no_grad():
            # get latent vector from encoder object (unsqueeze it to feed into LSTM)
            latent = self.encoder(image).unsqueeze(1)                             # size = (1, 1, embed_size)
            # initialize hidden states to be None (short_term, long_term)
            hidden_states = None
            # get hidden states after t = 0
            short_term, hidden_states = self.decoder.lstm(latent, hidden_states)  # size = (1, 1, hidden_size)
            # get output at t = 0 (squeeze dimension 1)
            output = self.decoder.linear(short_term).squeeze(1)                   # size = (1, vocab_size)
            # get probabilities for each word                                  
            prob_outputs = F.log_softmax(output[0], dim=0)                        # size = (vocab_size,)
            # choose top beam_size predictions
            _, idx = torch.topk(prob_outputs, beam_size)
            # initialize beams
            prev_beam = []
            next_beam = []
            for i in idx:
                prev_beam.append(([i], prob_outputs[i], hidden_states))
            # list of resulting captions (plausible)
            resulting_captions = []
            # generate captions (upto max_length length)
            for _ in range(max_length):
                # go through all captions till now (prev_beam)
                for word_list, prob, hidden_state in prev_beam:
                    # feed last word into LSTM (no teacher forcing)
                    last_word = self.decoder.embed(word_list[-1]).unsqueeze(0).unsqueeze(0)      # size = (1, 1, embed_size)
                    short_term, hidden_state = self.decoder.lstm(last_word, hidden_state)        # size = (1, 1, hidden_size)
                    # get word probabilities
                    prob_outputs = F.log_softmax(self.decoder.linear(short_term.squeeze(1))[0], dim=0) # size = (vocab_size,)
                    # choose top beam size_captions
                    _, idx = torch.topk(prob_outputs, beam_size)
                    # update next_beam
                    for i in idx:
                        next_beam.append((word_list + [i], prob + prob_outputs[i], hidden_state))
                # select top beam_size from beam_size * beam_size entries (sort in descending order of log-likelihood)
                next_beam.sort(reverse=True, key=lambda x: x[1])
                # filter next_beam
                prev_beam = []
                counter = 0
                for word_list, prob, hidden_state in next_beam:
                    if vocabulary[word_list[-1].item()] == '[END]':
                        # complete caption, store in resulting_captions
                        resulting_captions.append((word_list, prob))
                    else:
                        # append it into prev_beam
                        prev_beam.append((word_list, prob, hidden_state))
                        counter += 1
                    # beam_size captions stored, so break
                    if counter == beam_size:
                        break
                # reset next_beam
                next_beam = []
                # break if no more captions left
                if prev_beam == []:
                    break
            # sort captions in decreasing order of their log-likelihood
            resulting_captions.sort(reverse=True, key=lambda x: x[1])
            # get and return top caption
            caption = resulting_captions[0][0] if resulting_captions != [] else []
            if caption == []:
                return ['[START]', '[END]']
            else:
                return [vocabulary[idx.item()] for idx in caption]

# image encoder class (with attention module)
class EncoderAttn(nn.Module):
    # constructor for class EncoderAttn
    # batch_size: batch size during mini-batch optimization
    # map_size: number of pixels in output image (output of CNN)
    # feature_size: number of features in output image (output of CNN)
    def __init__(self, batch_size, map_size, feature_size):
        super(EncoderAttn, self).__init__()
        # set batch size (for mini-batch optimizations)
        self.batch_size = batch_size
        # set map size
        self.map_size = map_size
        # set feature size
        self.feature_size = feature_size
        # describe the CNN architecture
        # Key Highlights:
        # 1) Used 8 convolutional layers and a maximum of 512 kernel maps per layer
        # 2) Used ReLU as the activation function to avoid vanishing gradient and have faster convergence
        # 3) Used Max-Pooling after every 1-3 convolutional layers for scalability, robustness and to reduce noise (overfitting)
        # 4) Used batch normalization to standardize layer input (faster convergence) and provide some regularization (avoid overfitting)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(512, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        ) 
    # forward function for class EncoderAttn (perform forward propagation)
    # image: batch of image tensor (size = (batch_size, 3, 256, 256))
    def forward(self, image):
        # get the 3D feature map from image
        feature_map = self.cnn(image)                       # size = (batch_size, 512, 10, 10)
        # permute and view the feature map in appropriate way (for attention module)
        feature_map = feature_map.permute(0, 2, 3, 1).view(self.batch_size, self.map_size, self.feature_size)
        return feature_map                                  # size = (batch_size, 100, 512)

# attention module (Bahdanau Attention Mechanism)
class Attention(nn.Module):
    # constructor for class Attention
    # map_size: number of pixels in the CNN output (image)
    # feature_size: number of features in the CNN output (image)
    # hidden_size: size of the hidden state output of LSTM cell
    # attention_size: size of hidden vector in attention mechanism (generally equal to hidden_size)
    def __init__(self, map_size, feature_size, hidden_size, attention_size):
        super(Attention, self).__init__()
        # set map size
        self.map_size = map_size
        # set feature size
        self.feature_size = feature_size
        # set hidden size
        self.hidden_size = hidden_size
        # set attention size
        self.attention_size = attention_size
        # define the Linear layer (encoder)
        # Key Highlights: Converts encoder output (size = feature size) into a hidden vector (size = attention_size)
        self.encoder_linear = nn.Linear(feature_size, attention_size)
        # define the Linear layer (decoder)
        # Key Highlights: Converts decoder hidden state (size = hidden size) into a hidden vector (size = attention_size)
        self.decoder_linear = nn.Linear(hidden_size, attention_size)
        # define the Linear layer (weights)
        # Key Highlights: 
        # 1) Converts the activated hidden vector into 1-dimensional context-weight (attention weights)
        # 2) Tanh function is used for activation before determining the context-weight (Bahdanau)
        self.linear = nn.Linear(attention_size, 1)
        self.tanh = nn.Tanh()
    # forward function for class Attention
    # feature_map: encoder output (size = (batch_size, map_size, feature_size))
    # hidden_state: hidden state vector of LSTM (size = (batch_size, hidden_size))
    def forward(self, feature_map, hidden_state):
        # convert encoder output into hidden vector
        encoder_attention = self.encoder_linear(feature_map)                            # size = (batch_size, map_size, attention_size)
        # convert decoder hidden state into hidden vector
        decoder_attention = self.decoder_linear(hidden_state)                           # size = (batch_size, attention_size))
        # add both hidden vectors (use broadcasting in case of decoder-attention)
        # activate the sum using Tanh function
        total_attention = self.tanh(encoder_attention + decoder_attention.unsqueeze(1)) # size = (batch_size, map_size, attention_size)
        # determine weights from activated hidden vector
        weight = self.linear(total_attention)                                           # size = (batch_size, map_size, 1)
        # determine score for each pixel in the encoder output image (softmax)
        score = F.log_softmax(weight, dim=1)                                            # size = (batch_size, map_size, 1)
        # return context vector by calculating weighted sum of every pixel (use score as weights)
        return torch.sum(feature_map * score, dim=1)                                    # size = (batch_size, feature_size)

# caption decoder class (with attention module)
class DecoderAttn(nn.Module):
    # constructor for class DecoderAttn
    # map_size: number of pixels in the CNN output (image)
    # feature_size: number of features in the CNN output (image)
    # embed_size; size of embeddings (LSTM input)
    # hidden_size: size of hidden state (LSTM)
    # vocab_size: size of vocabulary dictionary
    # attention_size: size of hidden vector in attention mechanism (generally equal to hidden_size)
    # batch_size: batch size during mini-batch optimization
    def __init__(self, map_size, feature_size, embed_size, hidden_size, vocab_size, attention_size, batch_size):
        super(DecoderAttn, self).__init__()
        # set map size
        self.map_size = map_size
        # set feature size
        self.feature_size = feature_size
        # set embedding size
        self.embed_size = embed_size
        # set hidden size
        self.hidden_size = hidden_size
        # set vocabulary size
        self.vocab_size = vocab_size
        # set attention size
        self.attention_size = attention_size
        # set batch size
        self.batch_size = batch_size
        # set device (cuda or cpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # define embedding layer
        # Key Highlights:
        # 1) Generates embeddings with random weights for every word in the vocabulary
        # 2) Treates index 0 as padding ('[PAD]' token)
        # 3) Aim is to learn these weights
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # define LSTM Cell architecture
        # Key Highlights:
        # 1) Takes input vector of size (embed_size + feature_size) and produces hidden state vectors of size hidden_size
        # 2) Models the RNN by taking inputs over time and sharing weights
        self.lstm = nn.LSTMCell(embed_size + feature_size, hidden_size)
        # define Linear layer
        # Key Highlights:
        # 1) Used to convert the hidden state output of LSTM at each time step into a vocabulary vector
        # 2) This vocabulary vector is thereafter used for loss computation and inference
        # 3) Used dropout layer with probability = 0.1 to avoid overfitting
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        # define attention module (using Attention Class)
        self.attention_module = Attention(map_size, feature_size, hidden_size, attention_size)
        # define Linear layer
        # Key Highlights:
        # 1) Maps the 3D feature map obtained from CNN into a 1D latent vector
        # 2) Used Tanh as the activation function (since LSTMCell also uses it)
        # 3) Used batch normalization to standardize input (faster convergence) and avoid overfitting 
        self.hidden = nn.Linear(feature_size * map_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.tanh = nn.Tanh()
    # forward function of class DecoderAttn   
    # feature_map: feature_map received from EncoderAttn (size = (batch_size, 100, 512))
    # captions: batch of image captions for current batch (size = (batch_size, max_length))
    # lengths: batch of caption lengths (excluding padding) (size = (batch_size,))
    def forward(self, feature_map, captions, lengths):
        # determine max caption length (actual length)
        max_length = torch.max(lengths)
        # generate caption embeddings
        embeddings = self.embed(captions)                                                       # size = (batch_size, max_length, embed_size)
        # pack the input sequence to remove paddings ('[PAD]' token)
        # lengths vector stores the actual lengths of individual captions, so we also remove the '[END]' token
        packed_sequence = RNN.pack_padded_sequence(embeddings, lengths - 1, batch_first=True)   # size = (sum(lengths) - batch_size, embed_size)
        # store total words = sum(lengths) - batch_size
        total_words, _ = packed_sequence[0].size()
        # define temporary features_ (used to generate hidden state of LSTM at time 0)
        features_ = feature_map.contiguous().view(self.batch_size, self.map_size * self.feature_size) # size = (batch_size, 51200)
        # define short term and long term hidden states of LSTM using features_
        # apply hidden layer and batch normalization, followed by Tanh activation
        short_term = self.tanh(self.batch_norm(self.hidden(features_)))                          # size = (batch_size, hidden_size)
        long_term = self.tanh(self.batch_norm(self.hidden(features_)))                           # size = (batch_size, hidden_size)
        # define output tensor to store the output of LSTM cell (at each time step)
        output = torch.zeros(total_words, self.vocab_size)
        output = output.to(self.device)
        # start pointer: points to the next word embedding to be fed into LSTM cell
        start = 0
        # generate captions for max_length - 1 time steps (excluding '[END]' token)
        # Time step: t
        for t in range(max_length - 1):
            # determine number of inputs (size of input) at time step 't'
            size = packed_sequence[1][t].item()
            # get word embeddings for input words
            word_input = packed_sequence[0][start:(start + size), :]                                  # size = (size, embed_size)
            # get image context vector from attention module (Bahdanau)
            context_vector = self.attention_module(feature_map[0:size, :, :], short_term[0:size, :])  # size = (size, feature_size)
            # concatenate word embeddings with context vector to form LSTM input vector at time step 't'
            input_t = torch.cat((word_input, context_vector), dim=1)                                  # size = (size, embed_size + feature_size)
            # pass input_t through LSTM cell and get hidden state vectors (short term and long term)
            short_term, long_term = self.lstm(input_t, (short_term[0:size, :], long_term[0:size, :])) # size = (size, hidden_size) 
            # store corresponding vocabulary output, by passing short_term through a dropout layer followed by a linear layer
            output[start:(start + size), :] = self.linear(self.dropout(short_term))                   # size = (size, vocab_size)
            # update start pointer
            start += size
        # return decoder output
        return output

# image captioning claass (with attention module)
class ImageCaptionsNetAttn(nn.Module):
    # constructor for class ImageCaptionsNetAttn
    # map_size: number of pixels in the CNN output (image)
    # feature_size: number of features in the CNN output (image)
    # embed_size; size of embeddings (LSTM input)
    # hidden_size: size of hidden state (LSTM)
    # vocab_size: size of vocabulary dictionary
    # attention_size: size of hidden vector in attention mechanism (generally equal to hidden_size)
    # batch_size: batch size during mini-batch optimization
    def __init__(self, map_size, feature_size, embed_size, hidden_size, vocab_size, attention_size, batch_size):
        super(ImageCaptionsNetAttn, self).__init__()
        # set map size
        self.map_size = map_size
        # set feature size
        self.feature_size = feature_size
        # set embedding size
        self.embed_size = embed_size
        # set hidden size
        self.hidden_size = hidden_size
        # set vocabulary size
        self.vocab_size = vocab_size
        # set attention size
        self.attention_size = attention_size
        # set batch size
        self.batch_size = batch_size
        # set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # instantiate encoder object
        self.encoder = EncoderAttn(batch_size, map_size, feature_size)
        # instantiate decoder object
        self.decoder = DecoderAttn(map_size, feature_size, embed_size, hidden_size, vocab_size, attention_size, batch_size)
    # forward function of class ImageCaptionsNetAttn
    # image: batch of image tensors (size = (batch_size, 3, 256, 256))
    # captions: batch of image captions (size = (batch_size, max_length))
    # lengths: batch of actual captions length (excluding padding)
    def forward(self, image, captions, lengths):
        # get feature map from image
        feature_map = self.encoder(image)
        # return decoder output
        return self.decoder(feature_map, captions, lengths)
    # caption generator for class ImageCaptionsNet (uses beam search)
    # image: input image (size = (1, 3, 256, 256))
    # vocabulary: reverse vocabulary dictionary (idx2word)
    # beam_size: beam size to be used during beam search (default 5)
    # max_length: maximum length of caption generated (default 10)
    def caption_generator(self, image, vocabulary, beam_size=5, max_length=10):
        # do not use memory for gradients (inference time)
        with torch.no_grad():
            # get feature_map from encoder object
            feature_map = self.encoder(image)
            # determine initial hidden state vectors
            features_ = feature_map.contiguous().view(1, self.map_size * self.feature_size)
            short_term = self.decoder.tanh(self.decoder.batch_norm(self.decoder.hidden(features_)))
            long_term = self.decoder.tanh(self.decoder.batch_norm(self.decoder.hidden(features_)))
            # define '[START]' token and find its embedding
            start_token = torch.tensor(self.vocab_size - 2).to(self.device)
            word_input = self.decoder.embed(start_token).unsqueeze(0)       
            # determine context vector at t = 0 (from attention module)
            context_vector = self.decoder.attention_module(feature_map, short_term)
            # concatenate word embedding with context vector to form LSTM input vector at time step '0'
            input_t = torch.cat((word_input, context_vector), dim=1)
            # get hidden state output from LSTM cell
            short_term, long_term = self.decoder.lstm(input_t, (short_term, long_term))
            # get vocabulary output
            output = self.decoder.linear(short_term)
            # get probabilities for each word
            prob_outputs = F.log_softmax(output[0], dim=0)
            # choose top beam_size predictions
            _, idx = torch.topk(prob_outputs, beam_size)
            # initialize beams
            prev_beam = []
            next_beam = []
            for i in idx:
                prev_beam.append(([i], prob_outputs[i], (short_term, long_term)))
            # list of resulting captions (plausible)
            resulting_captions = []
            # generate captions (upto max_length)
            for _ in range(max_length):
                # go through all captions till now (prev_beam)
                for word_list, prob, (short_term, long_term) in prev_beam:
                    # feed last word into LSTM (no teacher forcing)
                    word_input = self.decoder.embed(word_list[-1]).unsqueeze(0)
                    # determine context vector at current time step
                    context_vector = self.decoder.attention_module(feature_map, short_term)
                    # concatenate word embedding with context vector to form LSTM input vector at current time step
                    input_t = torch.cat((word_input, context_vector), dim=1)
                    # get hidden state output from LSTM cell
                    short_term_, long_term_ = self.decoder.lstm(input_t, (short_term, long_term))
                    # get word probabilities
                    prob_outputs = F.log_softmax(self.decoder.linear(short_term_)[0], dim=0)
                    # choose top beam size_captions
                    _, idx = torch.topk(prob_outputs, beam_size)
                    # update next_beam
                    for i in idx:
                        next_beam.append((word_list + [i], prob + prob_outputs[i], (short_term_, long_term_)))
                # select top beam_size from beam_size * beam_size entries (sort in descending order of log-likelihood)
                next_beam.sort(reverse=True, key=lambda x: x[1])
                # filter next_beam
                prev_beam = []
                counter = 0
                for word_list, prob, (short_term, long_term) in next_beam:
                    if vocabulary[word_list[-1].item()] == '[END]':
                        # complete caption, store in resulting_captions
                        resulting_captions.append((word_list, prob))
                    else:
                        # append it into prev_beam
                        prev_beam.append((word_list, prob, (short_term, long_term)))
                        counter += 1
                    # beam_size captions stored, so break
                    if counter == beam_size:
                        break
                # reset next_beam
                next_beam = []
                # break if no more captions left
                if prev_beam == []:
                    break
            # sort captions in decreasing order of their log-likelihood
            resulting_captions.sort(reverse=True, key=lambda x: x[1])
            # get and return top caption
            caption = resulting_captions[0][0] if resulting_captions != [] else []
            if caption == []:
                return ['[START]', '[END]']
            else:
                return [vocabulary[idx.item()] for idx in caption]
