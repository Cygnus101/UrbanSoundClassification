# Urban Sound Classification

A lightweight pipeline for recognizing city noises (e.g. car horns, sirens, dog barks) on the UrbanSound8K dataset using MFCC‐CNN, CNN+LSTM, and fine-tuned transformer models.

## Installation

```bash
git clone https://github.com/Cygnus101/UrbanSoundClassification.git
cd UrbanSoundClassification
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# Unix/Mac
# source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --models/model.keras --input/audiofile.wav
```

##Citation

Pretrained Transformer
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}

Dataset
@techreport{salamon2014urbansound8k,
  title       = {UrbanSound8K: A Dataset of Urban Sounds},
  author      = {Salamon, Justin and Jacoby, Chris and Bello, Juan Pablo},
  institution = {New York University},
  year        = {2014},
  url         = {https://urbansounddataset.weebly.com/urbansound8k.html}
}

