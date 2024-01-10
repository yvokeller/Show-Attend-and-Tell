import argparse, json
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

# Global variable for tokenizer
global_tokenizer = {'tokenizer': None, 'bert': False}

def load_model(model_path=None, model_config_path=None, wandb_run=None, wandb_model=None):
    # Check if model is to be loaded from wandb
    if wandb_run is not None and wandb_model is not None:
        wandb_run_id = wandb_run.split('/')[2]
        wandb_model_config_name = wandb_model.split('/')[0] + '/model_config.json'
        model_target_dir = f'model/cache_wandb/{wandb_run_id}/'
        wandb_loaded_model = wandb.restore(name=wandb_model, run_path=wandb_run, root=model_target_dir)
        wandb_loaded_model_config = wandb.restore(name=wandb_model_config_name, run_path=wandb_run, root=model_target_dir)

        model_path = wandb_loaded_model.name
        model_config_path = wandb_loaded_model_config.name
    elif model_path is None or model_config_path is None:
        raise ValueError("Model path and config path must be provided if not loading from wandb")
    
    # Load model config
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)

    network = model_config['network']
    data_path = model_config['data']
    ado = model_config['ado']
    bert = model_config['bert']
    attention = model_config['attention']

    if bert:
        from transformers import BertTokenizer, BertModel 
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_tokenizer.bos_token = bert_tokenizer.cls_token
        bert_tokenizer.eos_token = bert_tokenizer.sep_token
        vocabulary_size = bert_model.config.vocab_size
        global_tokenizer['tokenizer'] = bert_tokenizer
        global_tokenizer['bert'] = True
    else:
        word_dict = json.load(open(data_path + '/word_dict.json', 'r'))
        vocabulary_size = len(word_dict)
        global_tokenizer['tokenizer'] = word_dict
        global_tokenizer['bert'] = False

    encoder = Encoder(network=network)
    decoder = Decoder(vocabulary_size, encoder.dim, ado=ado, bert=bert, attention=attention)
    try:
        decoder.load_state_dict(torch.load(model_path))
    except RuntimeError:
        print(f'Strict loading failed, loading with strict=False')
        decoder.load_state_dict(torch.load(model_path), strict=False)

    encoder.eval()
    decoder.eval()

    return encoder, decoder, bert, model_path, model_config_path

def generate_caption_visualization(img_path, encoder, decoder, model_config_path, model_path, beam_size=3, smooth=True, reload_tokenizer=False, figsize=None):
    if reload_tokenizer or global_tokenizer['tokenizer'] is None:
        encoder, decoder, bert = load_model(model_config_path, model_path)

    tokenizer = global_tokenizer['tokenizer']
    bert = global_tokenizer['bert']

    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha = decoder.caption(img_features, beam_size)

    if bert == True:
        # Decoding with BERT tokenizer
        sentence_tokens = tokenizer.decode(sentence, skip_special_tokens=False).split()
    else:
        # Decoding with custom word dictionary
        token_dict = {idx: word for word, idx in tokenizer.items()}
        sentence_tokens = []
        for word_idx in sentence:
            sentence_tokens.append(token_dict[word_idx])
            if word_idx == tokenizer['<eos>']:
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

    if figsize:
        plt.figure(figsize=figsize)

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

def main():
    parser = argparse.ArgumentParser(description='Show, Attend and Tell Caption Generator')
    parser.add_argument('--img-path', type=str, help='path to image')
    parser.add_argument('--model', type=str, help='path to model parameters')
    parser.add_argument('--wandb-run', type=str, help='wandb run path', default=None)
    parser.add_argument('--wandb-model', type=str, help='wandb model path', default=None)
    args = parser.parse_args()

    # Load the encoder and decoder models
    encoder, decoder, bert, model_path, model_config_path = load_model(args.model, None, args.wandb_run, args.wandb_model)

    # Generate caption visualization
    generate_caption_visualization(args.img_path, encoder, decoder, model_config_path, model_path)

if __name__ == "__main__":
    main()