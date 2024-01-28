import argparse, json
from enum import Enum
import os
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from PIL import Image
import skimage.transform
from os import cpu_count

from dataset import ImageCaptionDataset
from decoder import Decoder
from encoder import Encoder
from utils import AverageMeter, calculate_caption_lengths, count_parameters, sequence_accuracy, calculate_caption_lengths

import wandb

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

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
        bert_tokenizer.bos_token = bert_tokenizer.cls_token
        bert_tokenizer.eos_token = bert_tokenizer.sep_token
    else:
        word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
        vocabulary_size = len(word_dict)

    encoder = Encoder(args.network)
    decoder = Decoder(vocabulary_size, encoder.dim, tf=args.tf, ado=args.ado, bert=args.bert, attention=args.attention)

    if args.model:
        print(f"Fine-tuning from base model {args.model}")
        decoder.load_state_dict(torch.load(args.model))

    encoder.to(device)
    decoder.to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    start_time = time.time()
    train_loader = DataLoader(
        ImageCaptionDataset(data_transforms, args.data, fraction=args.fraction, bert=args.bert, split_type='train'),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    end_time = time.time()
    print(f"Time to load train dataset: {end_time - start_time} seconds")

    val_loader = DataLoader(
        ImageCaptionDataset(data_transforms, args.data, fraction=args.fraction, bert=args.bert, split_type='val'),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    test_loader = DataLoader(
        ImageCaptionDataset(data_transforms, args.data, fraction=args.fraction, bert=args.bert, split_type='test'),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print(f'Starting training with {args}')
    print("Encoder parameters:")
    count_parameters(encoder)
    print("Decoder parameters:")
    count_parameters(decoder)
    for epoch in range(1, args.epochs + 1):
        train(epoch, encoder, decoder, optimizer, cross_entropy_loss,
              train_loader, word_dict, args.alpha_c, args.log_interval, bert=args.bert, tokenizer=bert_tokenizer, args=args)
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


def train(epoch, encoder, decoder, optimizer, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, bert=False, tokenizer=None, args={}):
    print(f"Epoch {epoch} - Starting train")

    encoder.eval()
    decoder.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (imgs, captions, _) in enumerate(data_loader):
        # TODO: Lots of shared code with run_evaluation(), refactor?
        imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)

        img_features = encoder(imgs)
        optimizer.zero_grad()
        preds, alphas = decoder(img_features, captions)
        targets = captions[:, 1:] # skip <start> token for loss calculation

        # NOTE: for debugging
        # Assuming preds is a PyTorch tensor
        # if batch_idx == 0:
        #    torch.save(preds, f'preds_scenario1_epoch{epoch}.pt')

        # Calculate accuracy
        padding_idx = word_dict['<pad>'] if bert == False else tokenizer.pad_token_id
        acc1 = sequence_accuracy(preds, targets, 1, ignore_index=padding_idx, tokenizer=tokenizer)
        acc5 = sequence_accuracy(preds, targets, 5, ignore_index=padding_idx, tokenizer=tokenizer)

        # Calculate loss
        # remove paddings to avoid calculating loss on them - pack_padded_sequence() takes (data, lengths) as input, -1 because we skip <start> token
        # TODO: Currently PAD tokens are NOT removed from the loss calculation, but they are from the accuracy calculation
        packed_targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        # encourage total attention (alphas) to be close to 1, thus penalize when sum is far from 1
        att_regularization = alpha_c * ((1 - alphas.sum(1)) ** 2).mean()

        # add repetition penalty to loss
        # start_idx = word_dict['<start>'] if bert == False else tokenizer.cls_token_id
        # rep_penalty = repetition_penalty(preds, [padding_idx, start_idx], beta=1.0)
        # loss += rep_penalty

        loss = cross_entropy_loss(packed_preds, packed_targets)
        loss += att_regularization # pytorch autograd will calculate gradients for both loss and att_regularization
        loss.backward()
        optimizer.step()

        # NOTE: for debugging
        # Check if the gradients of the attention layer are None
        # if not args.attention:
        # for name, param in decoder.named_parameters():
        #    if 'attention' in name:
        #        if param.grad is not None:
        #            print(f"Gradients for {name} were computed.")

        if bert == True:
            total_caption_length = calculate_caption_lengths(captions, torch.tensor([tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id], device='mps'))
        else:
            total_caption_length = calculate_caption_lengths(captions, torch.tensor([word_dict['<pad>'], word_dict['<start>'], word_dict['<eos>']], device='mps'))

        losses.update(loss.item(), total_caption_length)
        top1.update(acc1, total_caption_length)
        top5.update(acc5, total_caption_length)

        if batch_idx % log_interval == 0:
            print(f'Train Batch: [{batch_idx}/{len(data_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  f'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})')

        wandb.log({
            'train_loss': losses.avg, 'train_top1_acc': top1.avg, 'train_top5_acc': top5.avg, 'epoch': epoch,
            'train_loss_raw': losses.val, 'train_top1_acc_raw': top1.val, 'train_top5_acc_raw': top5.val
        })

class EvalMode(Enum):
    VALIDATION = 'val'
    TEST = 'test'

def run_evaluation(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, bert=False, tokenizer=None, mode=EvalMode.VALIDATION):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    decoded_captions = [] # list of single assigned caption for each image
    decoded_all_captions = [] # list of list of all captions present in dataset for each image, thus captions may repeat in different lists
    decoded_hypotheses = [] # list of single predicted caption for each image
    
    predictions_table = wandb.Table(columns=["epoch", "mode", "target_caption", "pred_caption"])

    with torch.no_grad():
        logged_attention_visualizations_count = 0
        for batch_idx, (imgs, captions, all_captions) in enumerate(data_loader):
            imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)
            img_features = encoder(imgs)
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]

            # Calculate accuracy
            padding_idx = word_dict['<pad>'] if bert == False else tokenizer.pad_token_id
            acc1 = sequence_accuracy(preds, targets, 1, ignore_index=padding_idx, tokenizer=tokenizer)
            acc5 = sequence_accuracy(preds, targets, 5, ignore_index=padding_idx, tokenizer=tokenizer)

            # Calculate loss
            packed_targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = alpha_c * ((1 - alphas.sum(1)) ** 2).mean()

            loss = cross_entropy_loss(packed_preds, packed_targets)
            loss += att_regularization

            # NOTE: unused for now
            # add repetition penalty to loss
            # start_idx = word_dict['<start>'] if bert == False else tokenizer.cls_token_id
            # rep_penalty = repetition_penalty(preds, [padding_idx, start_idx], beta=1.0)
            # loss += rep_penalty

            if bert == True:
                total_caption_length = calculate_caption_lengths(captions, torch.tensor([tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id], device='mps'))
            else:
                total_caption_length = calculate_caption_lengths(captions, torch.tensor([word_dict['<pad>'], word_dict['<start>'], word_dict['<eos>']], device='mps'))
                
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            if bert == True:
                def bert_decode_caption(caption):
                    # NOTE: Trying new logic with stopping at <eos> token (relevant for BLEU score to represent later real world usage?)
                    tokens = tokenizer.convert_ids_to_tokens(caption)
                    sentence = []
                    for token in tokens:
                        if token == "[SEP]":
                            break
                        if token not in ("[CLS]", "[PAD]"):
                            sentence.append(token)

                    return tokenizer.convert_tokens_to_string(sentence).split()
                    # return tokenizer.decode(caption, skip_special_tokens=True).split()
                
                for caption in captions.tolist():
                    decoded_captions.append(bert_decode_caption(caption))

                for cap_set in all_captions.tolist():
                    caps = []
                    for caption in cap_set:
                        caps.append(bert_decode_caption(caption))
                    decoded_all_captions.append(caps)

                pred_captions = torch.max(preds, dim=2)[1]
                for pred_caption in pred_captions.tolist():
                    decoded_hypotheses.append(bert_decode_caption(pred_caption))
            else:
                token_dict = {idx: word for word, idx in word_dict.items()}
                def vanilla_decode_caption(caption):
                    # TODO: Trying new logic with stopping at <eos> token (relevant for BLEU score to represent later real world usage?)
                    sentence = []
                    for word_idx in caption:
                        if word_idx == word_dict['<eos>']:
                            break # Stop when EOS token is found
                        if word_idx not in (word_dict['<start>'], word_dict['<pad>']):
                            sentence.append(token_dict[word_idx])
                    return sentence
                    # return [token_dict[word_idx] for word_idx in caption if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]

                for caption in captions.tolist():
                    decoded_captions.append(vanilla_decode_caption(caption))

                for cap_set in all_captions.tolist():
                    caps = []
                    for caption in cap_set:
                        caps.append(vanilla_decode_caption(caption))
                    decoded_all_captions.append(caps)

                pred_captions = torch.max(preds, dim=2)[1]
                for pred_caption in pred_captions.tolist():
                    decoded_hypotheses.append(vanilla_decode_caption(pred_caption))

            if batch_idx % log_interval == 0:
                print(f'{mode} Batch: [{batch_idx}/{len(data_loader)}]\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                    f'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})')

            # In each batch, log the first image's predictions to W&B table
            predictions_table.add_data(epoch, mode.value, ' '.join(decoded_captions[-1]), ' '.join(decoded_hypotheses[-1]))
            
            if mode == EvalMode.TEST:
                # Calculate the start index for the current batch
                batch_start_idx = batch_idx * len(imgs)

                # Log the attention visualizations
                for img_idx, img_tensor in enumerate(imgs):
                    # Skip attention visualization if already logged enough
                    if logged_attention_visualizations_count >= 50:
                        break
                    logged_attention_visualizations_count += 1

                    # Calculate the global index for decoded_hypotheses and decoded_captions lists
                    global_caption_idx = batch_start_idx + img_idx

                    if len(decoded_hypotheses[global_caption_idx]) == 0:
                        print(f'No caption for image {global_caption_idx}, skipping attention visualization')
                        break

                    log_attention_visualization_plot(img_tensor, alphas, decoded_hypotheses, decoded_captions, batch_idx, img_idx, global_caption_idx, encoder)

        bleu_1 = corpus_bleu(decoded_all_captions, decoded_hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(decoded_all_captions, decoded_hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(decoded_all_captions, decoded_hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(decoded_all_captions, decoded_hypotheses)

        wandb.log({
            'epoch': epoch,
            f'{epoch}_{mode.value}_caption_predictions': predictions_table,
            f'{mode.value}_loss': losses.avg, f'{mode.value}_top1_acc': top1.avg, f'{mode.value}_top5_acc': top5.avg,
            f'{mode.value}_loss_raw': losses.val, f'{mode.value}_top1_acc_raw': top1.val, f'{mode.value}_top5_acc_raw': top5.val,
            f'{mode.value}_bleu1': bleu_1, f'{mode.value}_bleu2': bleu_2, f'{mode.value}_bleu3': bleu_3, f'{mode.value}_bleu4': bleu_4,
        })

        print(f'{mode} Epoch: {epoch}\t'
              f'BLEU-1 ({bleu_1})\t'
              f'BLEU-2 ({bleu_2})\t'
              f'BLEU-3 ({bleu_3})\t'
              f'BLEU-4 ({bleu_4})\t')

def validate(epoch, *args, **kwargs):
    print(f"Epoch {epoch} - Starting validation")
    return run_evaluation(epoch, *args, mode=EvalMode.VALIDATION, **kwargs)

def test(epoch, *args, **kwargs):
    print(f"Epoch {epoch} - Starting test")
    return run_evaluation(epoch, *args, mode=EvalMode.TEST, **kwargs)

def repetition_penalty(preds, ignore_idxs, beta=1.0):
    """
    Calculates a penalty for repetitions in the predictions.

    :param preds: Tensor of predicted token indices of shape [batch_size, seq_length, vocab_size].
    :param ignore_idxs: List of token indices to ignore for penalty calculation (e.g., padding or start tokens).
    :param beta: Weight for the repetition penalty.
    :return: Repetition penalty term.
    """
    # Get the predicted tokens (not probabilities)
    _, pred_tokens = preds.max(2)

    # Shift pred_tokens by one step
    shifted_pred_tokens = torch.cat((pred_tokens[:, :1], pred_tokens[:, :-1]), dim=1)

    # Calculate repetition (1 for same tokens, 0 for different)
    repetitions = (pred_tokens == shifted_pred_tokens).float()

    # Apply mask to ignore the specified token indices
    mask = torch.ones_like(repetitions).bool()
    for idx in ignore_idxs:
        mask &= (shifted_pred_tokens != idx)

    masked_repetitions = repetitions[:, 1:] * mask[:, 1:].float()

    # Sum the repetitions and average over batch size for penalty
    penalty = (masked_repetitions.sum() / pred_tokens.size(0)) * beta
    return penalty

def log_attention_visualization_plot(img_tensor, alphas, decoded_hypotheses, decoded_captions, batch_idx, img_idx, global_caption_idx, encoder):
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
    sentence_tokens = decoded_hypotheses[global_caption_idx]
    hypothesis_caption = ' '.join(decoded_hypotheses[global_caption_idx])
    reference_caption = ' '.join(decoded_captions[global_caption_idx])

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
    parser.add_argument('--attention', action='store_true', default=False,
                        help='use attention (default: False)')

    main(parser.parse_args())
