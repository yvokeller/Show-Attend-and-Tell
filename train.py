import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

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


def main(args):
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

    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, fraction=args.fraction, bert=args.bert),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, fraction=args.fraction, bert=args.bert, split_type='val'),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

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


def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, bert=False, tokenizer=None):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # used for calculating bleu scores
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
                for cap_set in all_captions.tolist():
                    caps = []
                    for caption in cap_set:
                        cap = [word_idx for word_idx in caption
                            if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                        caps.append(cap)
                    references.append(caps)

                word_idxs = torch.max(preds, dim=2)[1]
                for idxs in word_idxs.tolist():
                    hypotheses.append([idx for idx in idxs
                                    if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:
                print('Validation Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))

        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        wandb.log({
            'val_loss': losses.avg,
            'val_top1_acc': top1.avg,
            'val_top5_acc': top5.avg,
            'epoch': epoch,
            'val_bleu1': bleu_1,
            'val_bleu2': bleu_2,
            'val_bleu3': bleu_3,
            'val_bleu4': bleu_4
        })

        print('Validation Epoch: {}\t'
              'BLEU-1 ({})\t'
              'BLEU-2 ({})\t'
              'BLEU-3 ({})\t'
              'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))


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
