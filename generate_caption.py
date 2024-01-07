"""
We use the same strategy as the author to display visualizations
as in the examples shown in the paper. The strategy used is adapted for
PyTorch from here:
https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
"""

import argparse, json, os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
import torchvision.transforms as transforms
from math import ceil
from PIL import Image
import wandb

from dataset import pil_loader
from decoder import Decoder
from encoder import Encoder
from train import data_transforms

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")


def generate_caption_visualization(encoder, decoder, img_path, word_dict, beam_size=3, smooth=True, bert=False, tokenizer=None):
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha = decoder.caption(img_features, beam_size)

    if bert == True:
        # Decoding with BERT tokenizer
        sentence_tokens = tokenizer.decode(sentence, skip_special_tokens=True).split()
    else:
        # Decoding with custom word dictionary
        token_dict = {idx: word for word, idx in word_dict.items()}
        sentence_tokens = []
        for word_idx in sentence:
            sentence_tokens.append(token_dict[word_idx])
            if word_idx == word_dict['<eos>']:
                break

    img = Image.open(img_path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) / 2
    top = (h - 224) / 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
    img = img.astype('float32') / 255

    num_words = len(sentence_tokens)
    w = np.round(np.sqrt(num_words))
    h = np.ceil(np.float32(num_words) / w)
    alpha = torch.tensor(alpha)

    plot_height = ceil((num_words + 3) / 4.0)
    ax1 = plt.subplot(4, plot_height, 1)
    plt.imshow(img)
    plt.axis('off')
    for idx in range(num_words):
        ax2 = plt.subplot(4, plot_height, idx + 2)
        label = sentence_tokens[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, label, color='black', fontsize=13)
        plt.imshow(img)

        if encoder.network == 'vgg19':
            shape_size = 14
        else:
            shape_size = 7

        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alpha[idx, :].reshape(shape_size, shape_size), upscale=16,
                                                         sigma=20)
        else:
            alpha_img = skimage.transform.resize(alpha[idx, :].reshape(shape_size, shape_size),
                                                 [img.shape[0], img.shape[1]])
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    parser.add_argument('--img-path', type=str, help='path to image')
    parser.add_argument('--model', type=str, help='path to model parameters')
    parser.add_argument('--wandb-run', type=str, help='wandb run path', default=None)
    parser.add_argument('--wandb-model', type=str, help='wandb model path', default=None)
    args = parser.parse_args()
    
    # Load model from wandb
    if args.wandb_run is not None and args.wandb_model is not None:
        wandb_run_id = args.wandb_run.split('/')[2]
        wandb_model_config_name = args.wandb_model.split('/')[0] + '/model_config.json'
        model_target_dir = f'model/cache_wandb/{wandb_run_id}/'
        wandb_loaded_model = wandb.restore(name=args.wandb_model, run_path=args.wandb_run, root=model_target_dir)
        wandb_loaded_model_config = wandb.restore(name=wandb_model_config_name, run_path=args.wandb_run, root=model_target_dir)

        model_path = wandb_loaded_model.name
        model_config_path = wandb_loaded_model_config.name
    else:
        model_path = args.model
        model_config_path = os.path.join(os.path.dirname(model_path), 'model_config.json')

    # Load model config
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    network = model_config['network']
    data_path = model_config['data']
    ado = model_config['ado']
    bert = model_config['bert']

    if bert == True:
        from transformers import BertTokenizer, BertModel 
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        vocabulary_size = bert_model.config.vocab_size
    else:
        word_dict = json.load(open(data_path + '/word_dict.json', 'r'))
        vocabulary_size = len(word_dict)

    encoder = Encoder(network=network)
    decoder = Decoder(vocabulary_size, encoder.dim, ado=ado, bert=bert)
    decoder.load_state_dict(torch.load(model_path))

    encoder.eval()
    decoder.eval()

    if bert == True:
        generate_caption_visualization(encoder, decoder, args.img_path, None, bert=bert, tokenizer=bert_tokenizer)
    else:
        generate_caption_visualization(encoder, decoder, args.img_path, word_dict)
