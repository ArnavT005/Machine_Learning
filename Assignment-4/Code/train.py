import sys
import time
import torch
import random
import torch.nn as nn
from preprocessing import *
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ImageCaptionsNet, ImageCaptionsNetAttn

def main():
    # attention-flag
    attention = True if sys.argv[1] == '1' else False
    # degree of rotational invariance: 0, 1 or 2
    rotation = int(sys.argv[2])  
    # cut-time
    cut_time = int(sys.argv[3])
    # model-name (for saving)
    model_path = sys.argv[4]
    # data loading parameters (hard-coded)
    CAPTIONS_FILE_PATH = 'Train_text.tsv'
    IMAGE_DIR = 'processed_data'
    NUM_WORKERS = 8
    BATCH_SIZE = 16
    # load data
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)
    train_dataset = ImageCaptionsDataset(IMAGE_DIR, captions_preprocessing_obj.captions_dict, captions_preprocessing_obj.captions_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # select appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # set model parameters (hard-coded)
    embed_size = 512
    hidden_size = 512
    vocab_size = len(captions_preprocessing_obj.vocab.keys())
    if not attention:
        AlphaNet = ImageCaptionsNet(embed_size, hidden_size, vocab_size, BATCH_SIZE)
    else:
        map_size = 100
        feature_size = 512
        AlphaNet = ImageCaptionsNetAttn(map_size, feature_size, embed_size, hidden_size, vocab_size, hidden_size, BATCH_SIZE)
    # shift model to device and turn-on training mode
    AlphaNet = AlphaNet.to(device)
    AlphaNet.train()
    # set optimizer parameters
    # set global maximum number of epochs
    NUMBER_OF_EPOCHS = 1000
    # set loss function
    loss_function = nn.CrossEntropyLoss().to(device)
    # set optimizer (Adam)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, AlphaNet.parameters()))
    # start global timer
    global_timer = time.time()
    # training loop
    for epoch in range(NUMBER_OF_EPOCHS):
        start = time.time()
        # break if time exceeded
        if start - global_timer >= cut_time:
            break
        for iter, sample in enumerate(train_loader):
            # load captions, lengths
            captions_batch, lengths_batch = sample['captions'], sample['lengths']
            captions_batch = captions_batch.to(device)
            # sort lengths in descending order (required by pack_padded_sequence)
            lengths_batch, sequence = lengths_batch.sort(descending=True)
            captions_batch = captions_batch[sequence]
            # load images
            image_batch, image_90_batch, image_180_batch, image_270_batch = sample['image'], sample['image_90'], sample['image_180'], sample['image_270']
            image_batch, image_90_batch, image_180_batch, image_270_batch = image_batch.to(device), image_90_batch.to(device), image_180_batch.to(device), image_270_batch.to(device)
            image_batch, image_90_batch, image_180_batch, image_270_batch = image_batch.float()[sequence], image_90_batch.float()[sequence], image_180_batch.float()[sequence], image_270_batch.float()[sequence]
            # create image batches based on degree of rotational invariance
            if rotation == 0:
                # no rotation
                batches = [image_batch]
            elif rotation == 1:
                # random rotation
                batches = random.sample([image_batch, image_90_batch, image_180_batch, image_270_batch], 2)
            else:
                # full invariance
                batches = [image_batch, image_90_batch, image_180_batch, image_270_batch]
            # perform forward propagation and compute loss (perform backward propagation)
            if attention:
                for batch in batches:
                    captions_batch_ = captions_batch[:, 1:]
                    target_labels = nn.utils.rnn.pack_padded_sequence(captions_batch_, lengths_batch - 1, batch_first=True)[0]
                    target_labels = target_labels.to(device)
                    output_captions = AlphaNet(batch, captions_batch, lengths_batch)
                    output_captions = output_captions.to(device)
                    loss = loss_function(output_captions, target_labels)
                    # perform backward propagation
                    AlphaNet.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                for batch in batches:
                    target_labels = nn.utils.rnn.pack_padded_sequence(captions_batch, lengths_batch, batch_first=True)[0]
                    target_labels = target_labels.to(device)
                    output_captions = AlphaNet(batch, captions_batch, lengths_batch)
                    output_captions = output_captions.to(device)
                    loss = loss_function(output_captions, target_labels)
                    # perform backward propagation
                    AlphaNet.zero_grad()
                    loss.backward()
                    optimizer.step()
            # print statistics
            if iter % 10 == 0:
                end = time.time()
                print("Epoch:", epoch, "Iteration:", iter, ", Loss:", loss.item(), ", Time:", end - start)
    # save model
    torch.save(AlphaNet.state_dict(), model_path)

# invoke main()
if __name__ == '__main__':
    main()
