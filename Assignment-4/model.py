import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.AvgPool2d(3, 2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(True),
            nn.AvgPool2d(3, 2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(True),
            nn.AvgPool2d(3, 2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)  
        )
        self.linear = nn.Sequential(
            nn.Linear(512*9*9, 13824),
            nn.ReLU(True),
            nn.Linear(13824, self.embed_size),
            nn.ReLU(True)
            # nn.Dropout(p=0.5, inplace=False),
            # nn.Dropout(p=0.5, inplace=False),
            # nn.Dropout(p=0.5, inplace=False),
        )
    
    def forward(self, image):
        features = self.cnn(image)
        features = self.linear(features.view(features.size()[0], -1))
        return features

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed_seq)
        outputs = self.dropout(self.linear(hiddens[0]))  
        return outputs

class ImageCaptionsNet(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptionsNet, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.encoder = Encoder(embed_size)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, image, captions, lengths):
        features = self.encoder(image)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def caption_image_greedy(self, image, vocabulary, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(1)
            states = None
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                outputs = self.decoder.linear(hiddens.squeeze(1))
                predicted = outputs.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(1)
                if vocabulary[predicted.item()] == '[END]':
                    break
        return [vocabulary[idx] for idx in result_caption]
    
    def caption_image_beam_search(self, image, vocabulary, beam_size=10, max_length=50):
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(1)
            states = None
            hiddens, states = self.decoder.lstm(x, states)
            outputs = self.decoder.linear(hiddens.squeeze(1))
            prob_outputs = F.log_softmax(outputs[0], dim=0)
            values, idx = torch.topk(prob_outputs, beam_size)
            prev_beam = []
            next_beam = []
            resulting_captions = []
            # initialise beam
            for i in idx:
                prev_beam.append(([i], prob_outputs[i], states))
            for _ in range(max_length):
                for word_list, prob, hidden_state in prev_beam:
                    last_word = self.decoder.embed(word_list[-1]).unsqueeze(0).unsqueeze(0)
                    outs, hidden_state = self.decoder.lstm(last_word, hidden_state)
                    prob_outputs = F.log_softmax(self.decoder.linear(outs.squeeze(1))[0], dim=0)
                    values, idx = torch.topk(prob_outputs, beam_size)
                    for i in idx:
                        next_beam.append((word_list + [i], prob + prob_outputs[i], hidden_state))
                # select top beam_size from beam_size * beam_size entries
                next_beam.sort(reverse=True, key=lambda x: x[1])
                prev_beam = []
                counter = 0
                for word_list, prob, hidden_state in next_beam:
                    
                    if vocabulary[word_list[-1].item()] == '[END]':
                        resulting_captions.append((word_list, prob))
                    else:
                        prev_beam.append((word_list, prob, hidden_state))
                        counter += 1
                    if counter == beam_size:
                        break
                next_beam = []
                if prev_beam == []:
                    break
            resulting_captions.sort(reverse=True, key=lambda x: x[1])
            caption = resulting_captions[0][0] if resulting_captions != [] else []
            if caption == []:
                return ['[START]', '[END]']
            else:
                return [vocabulary[idx.item()] for idx in caption]
