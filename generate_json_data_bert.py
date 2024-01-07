import argparse
import json
from transformers import BertTokenizer

def generate_json_data(split_path, data_path, max_captions_per_image):
    split = json.load(open(split_path, 'r'))

    train_captions = []
    validation_captions = []
    test_captions = []

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Determine the maximum length of captions
    max_length = 0
    for img in split['images']:
        for sentence in img['sentences']:
            encoded_caption = tokenizer.encode(sentence['raw'], add_special_tokens=True)
            max_length = max(max_length, len(encoded_caption))

    for img in split['images']:
        caption_count = 0
        for sentence in img['sentences']:
            if caption_count < max_captions_per_image:
                caption_count += 1
            else:
                break

            # Tokenize each caption with BERT's tokenizer
            encoded_caption = tokenizer.encode(sentence['raw'], add_special_tokens=True)
            # Pad the caption to max_length
            padded_caption = encoded_caption + [tokenizer.pad_token_id] * (max_length - len(encoded_caption))

            if img['split'] == 'train':
                train_captions.append(padded_caption)
            elif img['split'] == 'val':
                validation_captions.append(padded_caption)
            elif img['split'] == 'test':
                test_captions.append(padded_caption)

    # Save the processed data
    with open(data_path + '/train_captions_bert.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_path + '/val_captions_bert.json', 'w') as f:
        json.dump(validation_captions, f)
    with open(data_path + '/test_captions_bert.json', 'w') as f:
        json.dump(test_captions, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json caption files for BERT tokenization')
    parser.add_argument('--split-path', type=str, default='data/coco/dataset.json')
    parser.add_argument('--data-path', type=str, default='data/coco')
    parser.add_argument('--max-captions', type=int, default=5, help='maximum number of captions per image')
    args = parser.parse_args()

    generate_json_data(args.split_path, args.data_path, args.max_captions)
