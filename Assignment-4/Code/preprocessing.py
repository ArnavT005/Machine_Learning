import os
import time
import torch
import shutil
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform, color

# class taken from starter code
# Key Highlights:
# 1) Splits the captions on spaces, generates index for every word
# 2) Adds [START], [END and [PAD] tokens to make all captions of equal length
# 3) Returns indexed tensor and actual lengths
class CaptionsPreprocessing:
    """Preprocess the captions, generate vocabulary and convert words to tensor tokens

    Args:
        captions_file_path (string): captions tsv file path
    """
    def __init__(self, captions_file_path):
        self.captions_file_path = captions_file_path
        # max caption length
        self.maxLen = 0
        # Read raw captions
        self.raw_captions_dict = self.read_raw_captions()
        # Preprocess captions
        self.captions_dict = self.process_captions()
        # Create vocabulary
        self.vocab = self.generate_vocabulary()
    def read_raw_captions(self):
        """
        Returns:
            Dictionary with raw captions list keyed by image ids (integers)
        """
        captions_dict = {}
        with open(self.captions_file_path, 'r', encoding='utf-8') as f:
            for img_caption_line in f.readlines():
                img_captions = img_caption_line.strip().split('\t')
                captions_dict[img_captions[0]] = img_captions[1]
                self.maxLen = max(self.maxLen, len(img_captions[1].split()) + 2)
        return captions_dict
    def process_captions(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """
        raw_captions_dict = self.raw_captions_dict
        # Do the preprocessing here
        captions_dict = {}
        # add START and END token
        for k, v in raw_captions_dict.items():
            captions_dict[k] = '[START] ' + v + ' [END]'
        return captions_dict
    def generate_vocabulary(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """
        captions_dict = self.captions_dict
        # Generate the vocabulary, add '[PAD]' token
        vocab = {'[PAD]': 0}
        idx = 1
        for caption in captions_dict.values():
            for word in caption.split():
                if word in ['[START]', '[END]']:
                    continue
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        # add '[START]' and '[END]' token
        vocab['[START]'] = idx
        vocab['[END]'] = idx + 1
        return vocab
    def captions_transform(self, img_caption):
        """
        Use this function to generate tensor tokens for the text captions
        Args:
            img_caption: caption for a particular image
        """
        vocab = self.vocab
        # Generate tensors
        tokens = [vocab[word] for word in img_caption.split()]
        length = len(tokens)
        tokens.extend([0 for _ in range(len(tokens), self.maxLen)])
        # return padded tensor and actual length
        return torch.tensor(tokens), length

# class taken from starter code
# Key Highlights:
# 1) Acts on processed images rather than raw images
# 2) For each image, it generates and stores all 4 rotations: used in making model rotational invariant
class ImageCaptionsDataset(Dataset):
    def __init__(self, img_dir, captions_dict, captions_transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            captions_dict: Dictionary with captions list keyed by image paths (strings)
            img_transform (callable, optional): Optional transform to be applied
                on the image sample.

            captions_transform: (callable, optional): Optional transform to be applied
                on the caption sample (list).
        """
        self.img_dir = img_dir
        self.captions_dict = captions_dict
        self.image_ids = list(captions_dict.keys())
        self.captions_transform = captions_transform
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        key = self.image_ids[idx]
        img_name = self.img_dir + key[key.index('/'):]
        # load and save all four possible rotations of the image
        image = io.imread(img_name)
        image_90 = transform.rotate(image, 90)
        image_180 = transform.rotate(image, 180)
        image_270 = transform.rotate(image, 270)
        # generate torch tensors (3, 256, 256)
        image = torch.tensor(image.transpose((2, 0, 1)))
        image_90 = torch.tensor(image_90.transpose((2, 0, 1)))
        image_180 = torch.tensor(image_180.transpose((2, 0, 1)))
        image_270 = torch.tensor(image_270.transpose((2, 0, 1)))
        captions = self.captions_dict[key]
        if self.captions_transform:
            captions, length = self.captions_transform(captions)
        sample = {'image': image, 'image_90': image_90, 'image_180': image_180, 'image_270': image_270, 'captions': captions, 'lengths': length}
        return sample

# function to process raw images and reduce background noise
# simple filtering is used to make processing faster and effective
# img_folder: image directory (relative)
# target_folder: target image directory (to be created) (relative)
def preprocess(img_folder, target_folder, threshold=(220/255)):
    # store path of img_folder
    # get all image paths in directory
    img_paths = os.listdir(img_folder)
    # make directory for processed images
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.mkdir(target_folder)
    # statistics
    start = time.time()
    print("Starting Image Processing...")
    img_counter = 0
    for img_path in img_paths:
        # read image
        img = io.imread(img_folder + "/" + img_path)
        # filter image
        img = color.rgb2gray(img)
        filter = (img >= threshold)
        img = img * filter
        # resize and convert image
        img = transform.resize(img, (256, 256), anti_aliasing=True)
        img = color.gray2rgb(img)
        # save as integer type
        img = img * 255
        img = img.astype(np.uint8)
        # save image in target folder
        io.imsave(target_folder + '/' + img_path, img, check_contrast=False)
        # statistics
        img_counter += 1
        end = time.time()
        if img_counter % 1000 == 0:
            print("Processed Image Count: " + str(img_counter) + ", Time (in s): " + str(end - start))
    print("Image Processing Complete")
