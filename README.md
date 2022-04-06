# AMR Parser for Bahasa Indonesia

AMR Parser from dependency parser features. Currently tested in Windows 10, not yet for Ubuntu.
## Getting Started

### Prerequisites

What things you need to install the software and how to install them

```
python 3.8.10 
tensorflow 2.5.0
conda (opt)
```

#### Optional

```bash
sudo apt update
sudo apt install -y tree unzip wget
```

### Installing

To install this app in your device, please do the following instructions. In case setting up for serving the model in limited memory, use python `venv` and install requirements using `--no-cache-dir` flag to avoid install process being killed because of low memory.

You can Install packages from the *requirements.txt* file

```
pip install -r requirements.txt
```

Alternatively, you can use conda instead, and create new environment from `environment.yml`

```
conda env create -f environment.yml
conda activate amr
```

then update anago package so that it use newest keras API, the following will download and replace the anago in python site packages directory, it will default on `venv/lib/py*/site-packages/anago` directory to search for the files.

```bash
chmod +x update-anago.sh
./update-anago.sh -d <anago site package folder>
```

2. Prepare stanza for Indonesian language support and Tools for evaluation

```bash
python prepare.py
```

Also prepare tools for evaluating AMR

```bash
git clone https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced tools/amr-eval
```

3. Download [pretrained features model](https://storage.googleapis.com/riset_amr/adylan/pretrained_feature_models.zip) that contains POS Tagger, NER Tagger, and Word2Vec models and put it in the `pretrained` folder. The tree structure should be like this:

``` bash
wget https://storage.googleapis.com/riset_amr/adylan/pretrained_feature_models.zip -O pretrained.zip && \
unzip pretrained.zip && \
rm pretrained.zip
```

The resulting directory tree will be:

```bash
$ tree pretrained/
pretrained
├── ner_tagger
│   ├── nerparams.json
│   ├── nerprepro.pkl
│   └── nerweight.h5
├── pos_tagger
│   └── all_indo_man_tag_corpus_model.crf.tagger
└── word2vec
    └── id
        ├── id.bin.gz
        └── id.tsv
```

4. For the current version, we'll be using new NER model to improve the entity detection. Download and extract it using the following
``` bash
wget https://storage.googleapis.com/riset_amr/pretrained_model/ner/model_ner_12514_softmax_v5_w2v_100_POS_LSTM_EmbNotTrainable_OOV-20210926T165506Z-001.zip
unzip model_ner_12514_softmax_v5_w2v_100_POS_LSTM_EmbNotTrainable_OOV-20210926T165506Z-001.zip -d pretrained
rm model_ner_12514_softmax_v5_w2v_100_POS_LSTM_EmbNotTrainable_OOV-20210926T165506Z-001.zip
```

for the current model we'll load it using the `load_model` of tensorflow to the pretrained model. `model_ner_12514_softmax_v5_w2v_100_POS_LSTM_EmbNotTrainable_OOV` also we'll load the pickle for vocabulary.

5. Download the [pretrained model and encoders](https://storage.googleapis.com/riset_amr/adylan/pretrained_model_and_encoder.zip) and copy the contents of the file to the `saved_model` directory.

``` bash
wget https://storage.googleapis.com/riset_amr/adylan/pretrained_model_and_encoder.zip -O saved_model.zip && \
unzip saved_model.zip -d saved_model && \
rm saved_model.zip
```

And will result in the following folder structure

``` bash
$ tree saved_model/
saved_model/
├── best_model_pretrained.pickle.dat
└── encoder
    ├── label_encoder.pickle.dat
    └── one_hot_encoder.pickle.dat
```


## Running the program

### Prediction

```
python amr_parser.py --predict --sentence "<any sentence that you want>" --output <filepath>
```

Example:
```
python amr_parser.py --predict --sentence "Aku makan buah di pagi ini" --output amr.txt
```

The output of the AMR should be included in the `amr.txt` file.

### Serving the Model for Prediction

For serving this model, we could use gunicorn. By first installing the package then run the server

```bash
pip install gunicorn
# gunicorn -w [NUM_WORKER] -b [ADDR:PORT] [PACKAGE]:app 
gunicorn -w 4 -b 0.0.0.0:5000 rest:app
#  or use -D to run it as daemon (kill with pkill gunicorn)
```

now we can test if it works or not from the browser or by using curl/wget
```bash
# wget http://[ADDR]:[PORT]/predict?sentence=[URL_ENCODED_TEXT] -O /dev/stdout -q
wget http://127.0.0.1:5000/predict?sentence=Ini%20adalah%20kalimat%20AMR -O /dev/stdout -q
```

If it works, next we can configure nginx and systemd around this in case of deployment to production.

### Training 

COMING SOON

## Authors

* **Adylan Roaffa Ilmy** 
* **Dr. Masayu Leylia Khodra, ST., MT.** - Advisor 


