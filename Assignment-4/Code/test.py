import os
import sys
import nltk
import torch
from skimage import io
from preprocessing import *
from model import ImageCaptionsNet, ImageCaptionsNetAttn

def main():
    # attention-flag
    attention = True if sys.argv[1] == '1' else False
    # model path (used for loading)
    model_path = sys.argv[2]
    # data loading parameters (hard-coded)
    CAPTIONS_FILE_PATH = 'Train_text.tsv'
    IMAGE_DIR = 'processed_data'
    # load data
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)
    # create reverse dictionary
    idx2word = {}
    for key, value in captions_preprocessing_obj.vocab.items():
        idx2word[value] = key
    # select appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # set model parameters (hard-coded)
    embed_size = 512
    hidden_size = 512
    vocab_size = len(captions_preprocessing_obj.vocab.keys())
    if not attention:
        AlphaNet = ImageCaptionsNet(embed_size, hidden_size, vocab_size, 1)
    else:
        map_size = 100
        feature_size = 512
        AlphaNet = ImageCaptionsNetAttn(map_size, feature_size, embed_size, hidden_size, vocab_size, hidden_size, 1)
    # load model from path
    AlphaNet.load_state_dict(torch.load(model_path, map_location=device))
    # shift model to device and turn-on evaluation mode
    AlphaNet = AlphaNet.to(device)
    AlphaNet.eval()
    # store path of img_folder
    directory = IMAGE_DIR
    bleu = 0
    iter = 0
    for img_path in captions_preprocessing_obj.captions_dict.keys():
        iter += 1
        # read image
        path = img_path[img_path.index('/'):]
        img = io.imread(directory + path)
        # convert to float tensor and shift to device
        img = torch.tensor(img.transpose((2, 0, 1)))
        img = img.unsqueeze(0)
        img = img.to(device).float()
        # determine output caption
        output_caption = AlphaNet.caption_generator(img, idx2word)
        actual_vector = captions_preprocessing_obj.captions_transform(captions_preprocessing_obj.captions_dict[img_path])
        # convert actual caption into words
        actual_caption = []
        for i in actual_vector[0]:
            if i.item() == 0:
                break
            if i.item() > vocab_size - 3:
                continue
            actual_caption.append(idx2word[i.item()])
        output_caption = [word for word in output_caption if word != '[START]' and word != '[END]']
        # create captions (join with spaces)
        candidate = " ".join(output_caption)
        reference = " ".join(actual_caption)
        # accumulate bleu score
        bleu += nltk.translate.bleu_score.sentence_bleu([list(reference)], list(candidate))
        if iter % 1000 == 0:
            print("Captions Predicted:", iter)
    bleu /= len(captions_preprocessing_obj.captions_dict.keys())
    print("BLEU Score:", bleu)

# invoke main()
if __name__ == '__main__':
    main()