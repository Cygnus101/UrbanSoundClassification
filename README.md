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

## References  

* Gong, Y., Chung, Y-A., & Glass, J. (2021). **AST: Audio Spectrogram Transformer**. *Proc. Interspeech 2021*, 571-575. https://doi.org/10.21437/Interspeech.2021-698  
* Salamon, J., Jacoby, C., & Bello, J. P. (2014). **UrbanSound8K** [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.1203745

