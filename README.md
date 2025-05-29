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
