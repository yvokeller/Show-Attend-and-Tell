import argparse, json
from enum import Enum
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from PIL import Image
import skimage.transform

from dataset import ImageCaptionDataset
from decoder import Decoder
from encoder import Encoder
from utils import AverageMeter, accuracy, calculate_caption_lengths, calculate_caption_lengths_bert

import wandb

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed) # for GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # for convolution operations
    torch.backends.cudnn.benchmark = False  # set to True if the input size does not vary for increased performance

def main(args):
    set_seed(args.seed)
    wandb.init(project='show-attend-and-tell', entity='yvokeller', config=args)

    bert_tokenizer = None
    word_dict = None
    if args.bert == True:
        from transformers import BertTokenizer, BertModel 
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        vocabulary_size = bert_model.config.vocab_size
    else:
        word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
        vocabulary_size = len(word_dict)

    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim, tf=args.tf, ado=args.ado, bert=args.bert)

    if args.model:
        decoder.load_state_dict(torch.load(args.model))

    encoder.to(mps_device)
    decoder.to(mps_device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    cross_entropy_loss = nn.CrossEntropyLoss().to(mps_device)

    # Load train data
    train_loader = DataLoader(
        ImageCaptionDataset(data_transforms, args.data, fraction=args.fraction, bert=args.bert),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    # Load validation and test data
    full_val_dataset = ImageCaptionDataset(data_transforms, args.data, fraction=args.fraction, bert=args.bert, split_type='val')
    if args.perform_test == True:
        test_fraction = 0.1

        # Determine the sizes for split
        val_size = int((1 - test_fraction) * len(full_val_dataset))
        test_size = len(full_val_dataset) - val_size

        # Split the full validation dataset into validation and test datasets
        val_dataset, test_dataset = random_split(full_val_dataset, [val_size, test_size])

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    else:
        val_dataset = full_val_dataset
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # Check for duplicates in full validation dataset
    img_path_set = set()
    for img_path in full_val_dataset.img_paths:
        if img_path in img_path_set:
            print(f"Duplicate found: {img_path}")
        else:
            img_path_set.add(img_path)

    # Check for duplicates between validation and test splits
    test_indices = set(test_dataset.indices)
    val_indices = set(val_dataset.indices)
    intersection = test_indices.intersection(val_indices)
    assert len(intersection) == 0, "There are duplicated indices in test and validation datasets!"

    print('Starting training with {}'.format(args))
    for epoch in range(1, args.epochs + 1):
        train(epoch, encoder, decoder, optimizer, cross_entropy_loss,
              train_loader, word_dict, args.alpha_c, args.log_interval, bert=args.bert, tokenizer=bert_tokenizer)
        validate(epoch, encoder, decoder, cross_entropy_loss, val_loader,
                 word_dict, args.alpha_c, args.log_interval, bert=args.bert, tokenizer=bert_tokenizer)
        scheduler.step()

        # Save model and log to W&B
        model_file = f'model/model_{args.network}_{epoch}.pth'
        torch.save(decoder.state_dict(), model_file)
        wandb.save(model_file)

        # Save model config
        with open('model/model_config.json', 'w') as f:
            json.dump(vars(args), f)
        wandb.save('model/model_config.json')

    if args.perform_test == True:
        test(epoch, encoder, decoder, cross_entropy_loss, test_loader,
             word_dict, args.alpha_c, args.log_interval, bert=args.bert, tokenizer=bert_tokenizer)

    wandb.finish()


def train(epoch, encoder, decoder, optimizer, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, bert=False, tokenizer=None):
    encoder.eval()
    decoder.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (imgs, captions) in enumerate(data_loader):
        imgs, captions = Variable(imgs).to(mps_device), Variable(captions).to(mps_device)

        img_features = encoder(imgs)
        optimizer.zero_grad()
        preds, alphas = decoder(img_features, captions)
        targets = captions[:, 1:]

        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        att_regularization = alpha_c * ((1 - alphas.sum(1)) ** 2).mean()

        loss = cross_entropy_loss(preds, targets)
        loss += att_regularization
        loss.backward()
        optimizer.step()

        if bert == True:
            total_caption_length = calculate_caption_lengths_bert(captions, tokenizer)
        else:
            total_caption_length = calculate_caption_lengths(captions, word_dict)
        acc1 = accuracy(preds, targets, 1)
        acc5 = accuracy(preds, targets, 5)
        losses.update(loss.item(), total_caption_length)
        top1.update(acc1, total_caption_length)
        top5.update(acc5, total_caption_length)

        if batch_idx % log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))

        wandb.log({'train_loss': losses.avg, 'train_top1_acc': top1.avg, 'train_top5_acc': top5.avg, 'epoch': epoch})

class EvaluationMode(Enum):
    VALIDATION = 'val'
    TEST = 'test'

def run_evaluation(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, bert=False, tokenizer=None, mode=EvaluationMode.VALIDATION):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    references = []
    hypotheses = []
    with torch.no_grad():
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):
            imgs, captions = Variable(imgs).to(mps_device), Variable(captions).to(mps_device)
            img_features = encoder(imgs)
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]

            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = alpha_c * ((1 - alphas.sum(1)) ** 2).mean()

            loss = cross_entropy_loss(packed_preds, targets)
            loss += att_regularization

            if bert == True:
                total_caption_length = calculate_caption_lengths_bert(captions, tokenizer)
            else:
                total_caption_length = calculate_caption_lengths(captions, word_dict)
                
            acc1 = accuracy(packed_preds, targets, 1)
            acc5 = accuracy(packed_preds, targets, 5)
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            if bert == True:
                for cap_set in all_captions.tolist():
                    caps = []
                    for caption in cap_set:
                        # Decode each caption to text and split into words
                        cap_text = tokenizer.decode(caption, skip_special_tokens=True)
                        caps.append(cap_text.split())
                    references.append(caps)

                word_idxs = torch.max(preds, dim=2)[1]
                for idxs in word_idxs.tolist():
                    # Decode each predicted sequence to text and split into words
                    hyp_text = tokenizer.decode(idxs, skip_special_tokens=True)
                    hypotheses.append(hyp_text.split())
            else:
                token_dict = {idx: word for word, idx in word_dict.items()}
                for cap_set in all_captions.tolist():
                    caps = []
                    for caption in cap_set:
                        cap = [token_dict[word_idx] for word_idx in caption if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                        caps.append(cap)
                    references.append(caps)

                word_idxs = torch.max(preds, dim=2)[1]
                for idxs in word_idxs.tolist():
                    hypotheses.append([token_dict[idx] for idx in idxs if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:
                print(f'{mode} Batch: [{batch_idx}/{len(data_loader)}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
            
            if mode == EvaluationMode.TEST:
                # Calculate the start index for the current batch
                batch_start_idx = batch_idx * len(imgs)

                # For the first image in the batch, log the attention visualization
                for img_idx, img_tensor in enumerate(imgs):
                    global_caption_idx = batch_start_idx + img_idx  # Calculate the global index for global references and hypotheses lists

                    if len(hypotheses[global_caption_idx]) == 0:
                        print(f'No caption for image {global_caption_idx}, skipping attention visualization')
                        break

                    log_attention_visualization_plot(img_tensor, alphas, hypotheses, references, batch_idx, img_idx, global_caption_idx, encoder)

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        wandb.log({
            f'{mode.value}_loss': losses.avg,
            f'{mode.value}_top1_acc': top1.avg,
            f'{mode.value}_top5_acc': top5.avg,
            'epoch': epoch,
            f'{mode.value}_bleu1': bleu_1,
            f'{mode.value}_bleu2': bleu_2,
            f'{mode.value}_bleu3': bleu_3,
            f'{mode.value}_bleu4': bleu_4
        })

        print(f'{mode} Epoch: {epoch}\t'
              'BLEU-1 ({bleu_1})\t'
              'BLEU-2 ({bleu_2})\t'
              'BLEU-3 ({bleu_3})\t'
              'BLEU-4 ({bleu_4})\t')

def validate(epoch, *args, **kwargs):
    print("Starting validation...")
    return run_evaluation(epoch, *args, mode=EvaluationMode.VALIDATION, **kwargs)

def test(epoch, *args, **kwargs):
    print("Starting test...")
    return run_evaluation(epoch, *args, mode=EvaluationMode.TEST, **kwargs)

def log_attention_visualization_plot(img_tensor, alphas, hypotheses, references, batch_idx, img_idx, global_caption_idx, encoder):
    # Move tensor to CPU and detach from the computation graph
    img_tensor = img_tensor.cpu().detach()
    alphas_tensor = alphas.cpu().detach()

    # Reversing the normalization in-place
    for t, m, s in zip(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)  

    # Transpose the tensor to PIL-friendly format (H, W, C)
    img_np = img_tensor.numpy().transpose(1, 2, 0)

    # Rescale to [0, 255] and convert to uint8
    img_np = (img_np * 255).astype('uint8')
    img_displayable = Image.fromarray(img_np)

    # Taking first reference and hypothesis captions
    sentence_tokens = hypotheses[global_caption_idx]
    hypothesis_caption = ' '.join(hypotheses[global_caption_idx])
    reference_caption = ' '.join(references[global_caption_idx][0])

    # Plot the attention map
    fig, axs = plt.subplots(1, len(sentence_tokens), figsize=(20, 10))
    if not isinstance(axs, np.ndarray): # If only one word, axs is not an array
        axs = [axs]
    for idx, word in enumerate(sentence_tokens):
        # Reshape the attention map to the image dimensions
        if encoder.network == 'vgg19':
            shape_size = 14
        else:
            shape_size = 7
        attention_map = skimage.transform.pyramid_expand(alphas_tensor[img_idx, idx].reshape(shape_size, shape_size), upscale=16, sigma=20)
        axs[idx].imshow(img_displayable)
        axs[idx].imshow(attention_map, cmap='gray', alpha=0.8)
        axs[idx].axis('off')
        axs[idx].text(0, 1, word, backgroundcolor='white', fontsize=13)
        axs[idx].text(0, 1, word, color='black', fontsize=13)
    plt.tight_layout()

    # Convert plot to image format and log it to W&B
    plt.savefig('temp_attention_plot.png')
    plt.close()
    plot_image = Image.open('temp_attention_plot.png')

    # Log the attention visualization
    wandb.log({
        f"Image B{batch_idx}-{img_idx}": [wandb.Image(img_displayable, caption=f"Hyp: {hypothesis_caption}\nRef: {reference_caption}"), wandb.Image(plot_image, caption="Attention Map")]
    })

    # Remove the temporary attention plot
    os.remove('temp_attention_plot.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--perform-test', action='store_true', default=True,
                        help='take frac 0.1 of validation data to perform test after training (default: True)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--data', type=str, default='data/coco',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='vgg19',
                        help='network to use in the encoder (default: vgg19)')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--tf', action='store_true', default=False,
                        help='use teacher forcing when training LSTM (default: False)')
    parser.add_argument('--ado', action='store_true', default=False,
                        help='use advanced deep output (default: False)')
    parser.add_argument('--fraction', type=float, default=1.0, metavar='F',
                        help='fraction of dataset to use (default: 1.0)')
    parser.add_argument('--bert', action='store_true', default=False,
                        help='use bert for word embeddings (default: False)')

    main(parser.parse_args())
