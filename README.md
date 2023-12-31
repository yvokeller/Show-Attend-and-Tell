# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

Built upon AaronCCWong's PyTorch [implementation](https://github.com/AaronCCWong/Show-Attend-and-Tell).

## A PyTorch implementation

For a trained model to load into the decoder, use

- [VGG19](https://www.dropbox.com/s/eybo7wvsfrvfgx3/model_10.pth?dl=0)
- [ResNet152](https://www.dropbox.com/s/0fptqsw3ym9fx2w/model_resnet152_10.pth?dl=0)
- [ResNet152 No Teacher Forcing](https://www.dropbox.com/s/wq0g2oo6eautv2s/model_nt_resnet152_10.pth?dl=0)
- [VGG19 No Gating Scalar](https://www.dropbox.com/s/li4390nmqihv4rz/model_no_b_vgg19_5.pth?dl=0)

### Some training statistics

BLEU scores for VGG19 (Orange) and ResNet152 (Red) Trained With Teacher Forcing.

| BLEU Score | Graph                        | Top-K Accuracy   | Graph                              |
|------------|------------------------------|------------------|------------------------------------|
| BLEU-1     | ![BLEU-1](/assets/bleu1.png) | Training Top-1   | ![Train TOP-1](/assets/top1.png)   |
| BLEU-2     | ![BLEU-2](/assets/bleu2.png) | Training Top-5   | ![Train TOP-5](/assets/top5.png)   |
| BLEU-3     | ![BLEU-3](/assets/bleu3.png) | Validation Top-1 | ![Val TOP-1](/assets/val_top1.png) |
| BLEU-4     | ![BLEU-4](/assets/bleu4.png) | Validation Top-5 | ![Val TOP-5](/assets/val_top5.png) |

## Setup before training

Follow the instructions for the dataset you choose to work with.

First, download Karpathy's data splits [here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits).

### Flickr8k

Download the Flickr8k images from [here](https://www.kaggle.com/datasets/adityajn105/flickr8k). Put the images in `data/flickr8k/imgs/`.
Place the Flickr8k data split JSON file in `data/flickr8k/`. It should be named `dataset.json`.

Run `python generate_json_data.py --split-path='data/flickr8k/dataset.json' --data-path='data/flickr8k'` to generate the JSON files needed for training.

If you want to use pre-trained BERT embeddings (`bert=True`), additionally run `python generate_json_data_bert.py --split-path='data/flickr8k/dataset.json' --data-path='data/flickr8k'` to generate the BERT-tokenized caption JSON files.

### COCO

Download the COCO dataset training and validation images. Put them in `data/coco/imgs/train2014` and `data/coco/imgs/val2014` respectively.
Put the COCO dataset split JSON file from Karpathy in `data/coco/`. It should be named `dataset.json`.

Run `python generate_json_data.py --split-path='data/coco/dataset.json' --data-path='data/coco'` to generate the JSON files needed for training.

## To Train

Start the training by running:

```bash
python train.py --data=data/flickr8k
```

or to make a small test run:

```bash
python train.py --data=data/flickr8k --tf --ado --epochs=1 --frac=0.02 --log-interval=2
```

The models will be saved in `model/` and the training statistics are uploaded to your W&B account.

My training statistics are available here: [W&B](https://wandb.ai/yvokeller/show-attend-and-tell)

## To Generate Captions

Note that together with the model parameters, a model_config.json is saved. This is required by `generate_caption.py` to properly load the model.

```bash
python generate_caption.py --img-path <PATH_TO_IMG> --model <PATH_TO_MODEL_PARAMETERS>
```

An example:

```bash
python generate_caption.py --img-path data/flickr8k/imgs/667626_18933d713e.jpg --model model/model_vgg19_5.pth
```

Working images:

- data/flickr8k/imgs/667626_18933d713e.jpg
- data/flickr8k/imgs/3718892835_a3e74a3417.jpg
- data/flickr8k/imgs/44856031_0d82c2c7d1.jpg

## Captioned Examples

### Correctly Captioned Images

![Correctly Captioned Image 1](/assets/tennis.png)

![Correctly Captioned Image 2](/assets/right_cap.png)

### Incorrectly Captioned Images

![Incorrectly Captioned Image 1](/assets/bad_cap.png)

![Incorrectly Captioned Image 2](/assets/wrong_cap.png)

## References

[Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf)

[Original Theano Implementation](https://github.com/kelvinxu/arctic-captions)

[Neural Machine Translation By Jointly Learning to Align And Translate](https://arxiv.org/pdf/1409.0473.pdf)

[Karpathy's Data splits](https://cs.stanford.edu/people/karpathy/deepimagesent/)
