import sys
import torch
import numpy as np
import pandas as pd
from skimage import io
from preprocessing import *
from model import ImageCaptionsNet, ImageCaptionsNetAttn

def main():
    # attention-flag
    attention = True if sys.argv[1] == '1' else False
    # model path (used for loading)
    model_path = sys.argv[2]
    # .tsv file which needs to be generated
    target_file = sys.argv[3]
    # number of test images
    num_images = int(sys.argv[4])
    # dictionary of predicted captions
    embeddings = {}
    # data load parameters (hard-coded)
    CAPTIONS_FILE_PATH = "Train_text.tsv"
    IMAGE_DIR = 'processed_test_data'
    # load data
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)
    # create reverse dictionary
    idx2word = {}
    for key, value in captions_preprocessing_obj.vocab.items():
        idx2word[value] = key    
    # select appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    embed_size = 512
    hidden_size = 512
    vocab_size = len(captions_preprocessing_obj.vocab.keys())
    if not attention:
        AlphaNet = ImageCaptionsNet(embed_size, hidden_size, vocab_size, 1)
    else:
        map_size = 100
        feature_size = 512
        AlphaNet = ImageCaptionsNetAttn(map_size, feature_size, embed_size, hidden_size, vocab_size, hidden_size, 1)
    # load model from model_path and shift to device (evaluation mode)
    AlphaNet.load_state_dict(torch.load(model_path, map_location=device))
    AlphaNet = AlphaNet.to(device)
    AlphaNet.eval()
    # store path of img_folder
    directory = IMAGE_DIR + '/test'
    for idx in range(num_images):
        # read image
        img = io.imread(directory + str(idx + 1) + ".jpg")
        # convert to float tensor and shift to device
        img = torch.tensor(img.transpose((2, 0, 1)))
        img = img.unsqueeze(0)
        img = img.to(device).float()
        # generate caption
        output_caption = AlphaNet.caption_generator(img, idx2word)
        # remove '[START]' and '[END]' tokens
        output_caption = [word for word in output_caption if word != '[START]' and word != '[END]']
        # create string from list (space-separated)
        embedding = " ".join(output_caption)
        # update embedding and likelihood if needed
        embeddings[idx] = embedding
        if embedding == "":
            print("EMPTY STRING")
    # create tsv file
    predictions = []
    for idx in range(num_images):
        predictions.append(['test_data/test' + str(idx + 1) + ".jpg", embeddings[idx]])
    # convert to data frame and save as tsv
    predictions = pd.DataFrame(np.array(predictions))
    predictions.to_csv(target_file, header=None, index=None, sep='\t')

# invoke main()
if __name__ == '__main__':
    main()