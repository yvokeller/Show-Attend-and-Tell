import argparse
import json
from transformers import BertTokenizer

def generate_json_data(split_path, data_path, max_captions_per_image):
    split = json.load(open(split_path, 'r'))

    train_img_paths = []
    train_captions = []
    validation_img_paths = []
    validation_captions = []

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for img in split['images']:
        caption_count = 0
        for sentence in img['sentences']:
            if caption_count < max_captions_per_image:
                caption_count += 1
            else:
                break

            try:
                filepath_defined = 'filepath' in img
            except KeyError:
                filepath_defined = False
            img_path = f"{data_path}/imgs{'/' + img['filepath'] if filepath_defined else ''}/{img['filename']}"

            # Tokenize each caption with BERT's tokenizer
            encoded_caption = tokenizer.encode(sentence['raw'], add_special_tokens=True)

            if img['split'] == 'train':
                train_img_paths.append(img_path)
                train_captions.append(encoded_caption)
            elif img['split'] == 'val':
                validation_img_paths.append(img_path)
                validation_captions.append(encoded_caption)

    # Save the processed data
    with open(data_path + '/train_img_paths_bert.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_path + '/val_img_paths_bert.json', 'w') as f:
        json.dump(validation_img_paths, f)
    with open(data_path + '/train_captions_bert.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_path + '/val_captions_bert.json', 'w') as f:
        json.dump(validation_captions, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate json files for BERT tokenization')
    parser.add_argument('--split-path', type=str, default='data/coco/dataset.json')
    parser.add_argument('--data-path', type=str, default='data/coco')
    parser.add_argument('--max-captions', type=int, default=5, help='maximum number of captions per image')
    args = parser.parse_args()

    generate_json_data(args.split_path, args.data_path, args.max_captions)
