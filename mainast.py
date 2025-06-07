#!/usr/bin/env python
"""
Single-file inference for the UrbanSound8K AST model
without downloading the 346 MB base checkpoint.

Usage
-----
python infer_local_ast.py \
       --wav  input/100648-1-0-0.wav \
       --ckpt models/ast_urbansound8k_finetuned.pth \
       --plot
"""

# ------------------------ 1. Imports & constants -----------------------------

import argparse
from pathlib import Path

import torch
import torchaudio
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoFeatureExtractor, ASTForAudioClassification



MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"   # backbone (config only)

ID2LABEL = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ------------------------ 2. Build model from config -------------------------

def load_model(ckpt_path: Path, device: str = "cpu") -> ASTForAudioClassification:
    """
    Create an AST model from its CONFIG (kB) and load local fine-tuned weights (∼100 MB).
    No hub weight download required.
    """
    cfg = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=False)  # tiny JSON
    cfg.num_labels = len(ID2LABEL)
    cfg.id2label = ID2LABEL
    cfg.label2id = LABEL2ID

    model = ASTForAudioClassification(cfg)          # empty weights
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)       # every key must match
    return model.to(device).eval()


# ------------------------ 3. Audio loader ------------------------------------

def load_mono_16k(wav_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    if wav.shape[0] > 1:                            # stereo → mono
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)                           # shape [T]


# ------------------------ 4. Inference routine -------------------------------

@torch.inference_mode()
def predict(wav_path: Path, ckpt_path: Path, top_k: int = 5,
            device: str = "cuda", plot: bool = False):
    
    wav_path  = Path(wav_path)     # ← convert string → Path
    ckpt_path = Path(ckpt_path) 
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    model = load_model(ckpt_path, device)

    waveform = load_mono_16k(wav_path)
    inputs = extractor(waveform.numpy(), sampling_rate=16_000,
                       return_tensors="pt", padding=True).to(device)

    logits = model(**inputs).logits.squeeze(0)
    probs = logits.softmax(dim=-1).cpu()

    # print top-k
    topk = probs.topk(top_k)
    print(f"\n► {wav_path.name}")
    for rank, (idx, score) in enumerate(zip(topk.indices, topk.values), 1):
        print(f"{rank:>2}. {ID2LABEL[int(idx)]:<15} — {score:.3f}")

    # optional bar plot
    if plot:
        plt.figure(figsize=(9, 3))
        plt.bar(ID2LABEL.values(), probs)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Probability")
        plt.title("Class probabilities")
        plt.tight_layout()
        plt.show()


# ------------------------ 5. CLI wrapper -------------------------------------

def main():

    parser = argparse.ArgumentParser(description="UrbanSound8K AST inference (offline)")
    parser.add_argument("--wav",  default="input/100648-1-0-0.wav")
    parser.add_argument("--ckpt", default="models/ast_urbansound8k_finetuned.pth")
    parser.add_argument("--top_k", type=int, default=5, help="Show top-k classes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true", help="Display probability bar plot")
    args = parser.parse_args()

    predict(args.wav, args.ckpt, args.top_k, args.device, args.plot)


if __name__ == "__main__":
    main()
