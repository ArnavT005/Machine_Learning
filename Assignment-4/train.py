from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import ImageCaptionsNet

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
        # Generate the vocabulary
        vocab = {'[PAD]': 0}
        idx = 1
        for caption in captions_dict.values():
            for word in caption.split():
                if word in ['[START]', '[END]']:
                    continue
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
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
        return torch.tensor(tokens), length

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
        self.captions_transform = captions_transform
        self.image_ids = list(captions_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_name = self.img_dir + img_name[img_name.index('/'):]
        image = io.imread(img_name)
        image = torch.tensor(image.transpose((2, 0, 1)))
        captions = self.captions_dict[img_name]
        if self.captions_transform:
            captions, length = self.captions_transform(captions)
        sample = {'image': image, 'captions': captions, 'lengths': length}
        return sample

def main():
    # Set the captions tsv file path
    CAPTIONS_FILE_PATH = 'Train_text.tsv'
    IMAGE_DIR = 'processed_data'
    NUMBER_OF_EPOCHS = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)

    train_dataset = ImageCaptionsDataset(IMAGE_DIR, captions_preprocessing_obj.captions_dict, captions_preprocessing_obj.captions_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)

    embed_size = 2048
    hidden_size = 2048
    vocab_size = len(captions_preprocessing_obj.vocab.keys())
    num_layers = 1
    
    AlphaNet = ImageCaptionsNet(embed_size, hidden_size, vocab_size, num_layers)
    AlphaNet = nn.DataParallel(AlphaNet)
    AlphaNet.to(device)
    AlphaNet.train()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(AlphaNet.parameters())
    
    iter = 0
    for _ in range(NUMBER_OF_EPOCHS):
        start = time.time()
        for _, sample in enumerate(train_loader):
            iter += 1
            end = time.time()

            captions_batch, lengths_batch = sample['captions'], sample['lengths']
            captions_batch.to(device)
            
            lengths_batch, sequence = lengths_batch.sort(descending=True)
            captions_batch = captions_batch[sequence]
            
            target_labels = torch.nn.utils.rnn.pack_padded_sequence(captions_batch, lengths_batch, batch_first=True)[0]

            image_batch = sample['image']
            image_batch.to(device)
            image_batch = image_batch.float()[sequence]
            

            output_captions = AlphaNet(image_batch, captions_batch, lengths_batch)
            loss = loss_function(output_captions, target_labels)

            AlphaNet.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("Iteration:", iter, ", Loss:", loss.item(), ", Time:", end - start)
            
            if device != "cpu":
                captions_batch.detach().cpu()
                image_batch.detach().cpu()


if __name__ == '__main__':
    main()