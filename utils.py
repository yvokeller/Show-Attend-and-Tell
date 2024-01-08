import torch


class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    # Flaws in this function:
    # 1. Padding Token Handling: This function does not account for padding tokens. If the dataset 
    #    contains padded sequences, this function may count predictions of padding tokens (often 0) 
    #    as correct, artificially inflating the accuracy. Accurate accuracy measurement requires 
    #    masking out these padding tokens.
    #
    # 2. Top-k Accuracy for k > 1: The function is not correctly set up to handle top-k accuracy 
    #    for k > 1. It assumes that the correct prediction must be in the first position of the top-k 
    #    predictions (pred). For k > 1, the function should check if the correct prediction is 
    #    anywhere within the top k predictions for each position in the sequence.
    #
    # 3. Batch Size Misinterpretation: If packed sequences were previously passed to this function,
    #    'batch_size' would incorrectly represent the total length of the packed sequences in the batch,
    #    not the actual number of sequences (images) in the batch. This misinterpretation leads to 
    #    incorrect normalization of the accuracy score.
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)

def sequence_accuracy(preds, targets, k, ignore_index=0):
    """
    Calculates the sequence accuracy at k, considering only non-padding tokens.
    
    :param preds: Predictions tensor of shape [batch_size, seq_length, vocab_size]
    :param targets: Targets tensor of shape [batch_size, seq_length]
    :param k: Top k predictions to consider
    :param ignore_index: Index to ignore for padding tokens
    :return: Accuracy at k as a percentage
    """
    # Get top-k predictions across the vocabulary dimension
    _, topk_preds = preds.topk(k, dim=2, largest=True, sorted=True)
    
    # Expand targets to match the shape of topk_preds for comparison
    targets_expanded = targets.unsqueeze(-1).expand_as(topk_preds)
    
    # Create a mask for non-padding tokens
    mask = targets.ne(ignore_index)
    
    # Compare predictions to targets, count correct ones while ignoring padding
    correct_k = topk_preds.eq(targets_expanded) # Returns [batch_size, seq_length, k]
    correct_k_masked = correct_k * mask.unsqueeze(-1) # Apply mask to ignore padding
    
    # Use .any() along the 'k' dimension to check if any of the top k predictions is correct
    correct_any_k = correct_k_masked.any(dim=2) # Returns [batch_size, seq_length]
    
    # Sum the correct predictions
    correct_total = correct_any_k.float().sum()
    
    # Calculate the total number of non-padding tokens
    total_non_padding = mask.sum()
    
    # Return accuracy as a percentage
    return (correct_total.item() * 100.0 / total_non_padding.item()) if total_non_padding.item() > 0 else 0

def calculate_caption_lengths(captions, word_dict):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths

def calculate_caption_lengths_bert(captions, tokenizer):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id):
                continue
            else:
                lengths += 1
    return lengths