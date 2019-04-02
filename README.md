


## CNN + seq2seq with visual attention

![enter image description here](https://cdn-images-1.medium.com/max/2600/1*_Nb5AADlqVQJDa0YyNFKGA.jpeg)

 ![enter image description here](https://dv-website.s3.amazonaws.com/uploads/2018/05/kf_ann_052418.png)![enter image description here](data:image/jpeg;base64,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iMr/AO8eK+athDJHsa8SBri0PbfdcAbBwvwOa6EHdR1b4ntkie5j2m7XNJBB6EKyGybaa2vaKaqcG1TRgchMBxHJ/MeIVaV2QTOY5r2OLXNILXNNiCMiDwKC76KMdk201te0U1UQ2qaMDgBMBxHJ/MeI42k5AREQEREBERAREQEREBERAREQEREBERAREQEREBeNrnDI+gq2RX33QShtsySw4DqvZRBR0hYU47Ydlnv19Czm6aFo8TJGPq3xCg5AREQdkEzmOa9ji1zSC1wNiCMiDzVktk20xte0U1SQ2qaMDkJgOI5P5jxHStK7IJnMcHscWuaQWuBsQRkQUF30UY7Jtpja9opakhtU0YHITAcRyfzHiOknICIiAiIgIiICIiAiIgIiICIiAiIgIiICLF1lAREQFB22HZZ79fQM5umhaPEyRj6t8QpxRBRxFOO2HZb79fQs5umhaPEyRj6t8QoOQEREHZBM5jmvY4tc0gtc02IIxBB4FWR2TbTW17RS1RDapowOAEwAzHJ/MeI5CtS7IJnMc17HFrmkFrgbEEYgg80F30UY7Jtpra9opqkhtU0YHITAcRyfzHiFJyAiIgIiICIiAiIgwSl1myxZBlERAREQEREBERAREQEREBQbtg2We/X0DObpoWjxMkY+rfEKckQUcRTjth2We/X0DObpoWjxMkY+rfEKDkBERB2QTOY4PY4tc0gtcDYgjIg81ZHZLtMbXtFLUkNqmjA5CYDiOT+Y8RxtWpSpsd2cyVcjK2feigjcHMsS18r2m43TmGA5njkEFj0REBERAREQEREBERAREQEREBERAREQEREBERAREQFBm2HZb79fQM5umhaPEyRj6t8QpzRBRxFOO2DZZ79dQM5umhaPEyRgebfELXtkuzJ1c4VVU0tpWn2WnAzEcB+DmeOQQY2S7MnVzm1VU0tpWm7QcDMRwH4OZ4qyEELWNDGNDWtAAAFgAMgByWYIWsaGMaGtaAAALAAZADgFzQEREBERAREQEREBERAREQEREBERAREQEREBERAREQEREGCvl0V/cx/uhEQfWiIgIiIP/9k=)


## Acknowledgements

This project is based on a model by [Qi Guo](http://qiguo.ml) and [Yuntian Deng](https://github.com/da03). You can find the original model in the [da03/Attention-OCR](https://github.com/da03/Attention-OCR) repository.

## The model

Authors: [Qi Guo](http://qiguo.ml) and [Yuntian Deng](https://github.com/da03).

The model first runs a sliding CNN on the image (images are resized to height 32 while preserving aspect ratio). Then an LSTM is stacked on top of the CNN. Finally, an attention model is used as a decoder for producing the final outputs.

![OCR example](http://cs.cmu.edu/~yuntiand/OCR-2.jpg)

## Installation

```
pip install aocr
```

Note: Tensorflow and Numpy will be installed as dependencies. Additional dependencies are `PIL`/`Pillow`, `distance`, and `six`.

## Usage

### Create a dataset

To build a TFRecords dataset, you need a collection of images and an annotation file with their respective labels.

```
aocr dataset ./datasets/annotations-training.txt ./datasets/training.tfrecords
aocr dataset ./datasets/annotations-testing.txt ./datasets/testing.tfrecords
```

Annotations are simple text files containing the image paths (either absolute or relative to your working dir) and their corresponding labels:

```
datasets/images/hello.jpg hello
datasets/images/world.jpg world
```

### Train

```
aocr train ./datasets/training.tfrecords
```

A new model will be created, and the training will start. Note that it takes quite a long time to reach convergence, since we are training the CNN and attention model simultaneously.

The `--steps-per-checkpoint` parameter determines how often the model checkpoints will be saved (the default output dir is `checkpoints/`).

**Important:** there is a lot of available training options. See the CLI help or the `parameters` section of this README.

### Test and visualize

```
aocr test ./datasets/testing.tfrecords
```

Additionally, you can visualize the attention results during testing (saved to `out/` by default):

```
aocr test --visualize ./datasets/testing.tfrecords
```

Example output images in `results/correct`:

Image 0 (j/j):

![example image 0](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_0.jpg)

Image 1 (u/u):

![example image 1](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_1.jpg)

Image 2 (n/n):

![example image 2](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_2.jpg)

Image 3 (g/g):

![example image 3](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_3.jpg)

Image 4 (l/l):

![example image 4](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_4.jpg)

Image 5 (e/e):

![example image 5](http://cs.cmu.edu/~yuntiand/2evaluation_data_icdar13_images_word_370.png/image_5.jpg)

### Export

After the model is trained and a checkpoint is available, it can be exported as either a frozen graph or a SavedModel.

```bash

# SavedModel (default):
aocr export ./exported-model

# Frozen graph:
aocr export --format=frozengraph ./exported-model

```

Load weights from the latest checkpoints and export the model into the `./exported-model` directory.

### Serving

Exported SavedModel can be served as a HTTP REST API using [Tensorflow Serving](https://github.com/tensorflow/serving). You can start the server by running following command:

```
tensorflow_model_server --port=9000 --rest_api_port=9001 --model_name=yourmodelname --model_base_path=./exported-model
```

**Note**: tensorflow_model_server requires a sub-directory with the version number to be present and inside it the files exported in the previous step. So you need to manually move contents of `exported-model` into `exported-model/1`.

Now you can send a prediction request to the running server, for example:

```
curl -X POST \
  http://localhost:9001/v1/models/yourmodelname:predict \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -d '{
  "signature_name": "serving_default",
  "inputs": {
     	"input": { "b64": "/9j/4AAQ==" }
  }
}'
```

REST API requires binary inputs to be encoded as Base64 and wrapped in an object containing `b64` key. [See 'Encoding binary values' in Tensorflow Serving documentation](https://www.tensorflow.org/serving/api_rest#encoding_binary_values)



## Google Cloud ML Engine

To train the model in the [Google Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/), upload the training dataset into a Google Cloud Storage bucket and start a training job with the `gcloud` tool.

1. Set the environment variables:

```
# Prefix for the job name.
export JOB_PREFIX="aocr"

# Region to launch the training job in.
# Should be the same as the storage bucket region.
export REGION="us-central1"

# Your storage bucket.
export GS_BUCKET="gs://aocr-bucket"

# Path to store your training dataset in the bucket.
export DATASET_UPLOAD_PATH="training.tfrecords"
```

2. Upload the training dataset:

```
gsutil cp ./datasets/training.tfrecords $GS_BUCKET/$DATASET_UPLOAD_PATH
```

3. Launch the ML Engine job:

```
export NOW=$(date +"%Y%m%d_%H%M%S")
export JOB_NAME="$JOB_PREFIX$NOW"
export JOB_DIR="$GS_BUCKET/$JOB_NAME"

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir=$JOB_DIR \
    --module-name=aocr \
    --package-path=aocr \
    --region=$REGION \
    --scale-tier=BASIC_GPU \
    --runtime-version=1.2 \
    -- \
    train $GS_BUCKET/$DATASET_UPLOAD_PATH \
    --steps-per-checkpoint=500 \
    --batch-size=512 \
    --num-epoch=20
```

## Parameters

### Global

* `log-path`: Path for the log file.

### Testing

* `visualize`: Output the attention maps on the original image.

### Exporting

* `format`: Format for the export (either `savedmodel` or `frozengraph`).

### Training

* `steps-per-checkpoint`: Checkpointing (print perplexity, save model) per how many steps
* `num-epoch`: The number of whole data passes.
* `batch-size`: Batch size.
* `initial-learning-rate`: Initial learning rate, note the we use AdaDelta, so the initial value does not matter much.
* `target-embedding-size`: Embedding dimension for each target.
* `attn-num-hidden`: Number of hidden units in attention decoder cell.
* `attn-num-layers`: Number of layers in attention decoder cell. (Encoder number of hidden units will be `attn-num-hidden`*`attn-num-layers`).
* `no-resume`: Create new weights even if there are checkpoints present.
* `max-gradient-norm`: Clip gradients to this norm.
* `no-gradient-clipping`: Do not perform gradient clipping.
* `gpu-id`: GPU to use.
* `use-gru`: Use GRU cells instead of LSTM.
* `max-width`: Maximum width for the input images. WARNING: images with the width higher than maximum will be discarded.
* `max-height`: Maximum height for the input images.
* `max-prediction`: Maximum length of the predicted word/phrase.

## References

[Convert a formula to its LaTex source](https://github.com/harvardnlp/im2markup)

[What You Get Is What You See: A Visual Markup Decompiler](https://arxiv.org/pdf/1609.04938.pdf)

[Torch attention OCR](https://github.com/da03/torch-Attention-OCR)
