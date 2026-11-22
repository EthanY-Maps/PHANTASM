"""Recognition accuracy of the fine-tuned TrOCR model on held-out labelled crops.

Section 7.3 reported Stage 3 through coverage and confidence proxies, which do not
establish recognition accuracy. This measures it directly: character and word error
rates and exact-match rate over the labelled test crops, case-sensitive and
case-insensitive, since the gazetteer match that consumes the output is case-blind.

Results are written to JSON as well as printed. The first pass only printed them, which
left the manuscript's numbers resting on transcribed stdout with nothing on disk to
regenerate them from.
"""
import json, os, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PHD = Path("data")
MODEL = Path(__file__).resolve().parent.parent / "models/trocr_street_names"   # paper-era TrOCR
TEST = PHD / "Training/OCR/Test_Images_Renamed"

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

files = sorted(f for f in os.listdir(TEST) if f.lower().endswith((".jpg", ".png")))
labels = [re.sub(r"_\d+$", "", os.path.splitext(f)[0]) for f in files]
print(f"model: {MODEL}")
print(f"crops: {len(files)}")

proc = TrOCRProcessor.from_pretrained(str(MODEL))
model = VisionEncoderDecoderModel.from_pretrained(str(MODEL)).eval()

preds = []
B = 16
for i in range(0, len(files), B):
    ims = [Image.open(TEST / f).convert("RGB") for f in files[i:i+B]]
    px = proc(images=ims, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(px, max_length=32)
    preds += proc.batch_decode(ids, skip_special_tokens=True)

def score(gt, pr):
    ce = sum(lev(g, p) for g, p in zip(gt, pr)); cn = sum(len(g) for g in gt)
    we = sum(lev(g.split(), p.split()) for g, p in zip(gt, pr)); wn = sum(len(g.split()) for g in gt)
    ex = sum(g == p for g, p in zip(gt, pr))
    return ce/cn, we/wn, ex/len(gt)

cer, wer, ex = score(labels, [p.strip() for p in preds])
cerL, werL, exL = score([l.lower() for l in labels], [p.strip().lower() for p in preds])
print(f"\n{'':<18}{'CER':>9}{'WER':>9}{'exact':>9}")
print(f"{'case-sensitive':<18}{cer:9.3f}{wer:9.3f}{ex:9.3f}")
print(f"{'case-insensitive':<18}{cerL:9.3f}{werL:9.3f}{exL:9.3f}")
print("\nsample (gt -> pred):")
for g, p in list(zip(labels, preds))[:10]:
    flag = "" if g.lower() == p.strip().lower() else "   <-- miss"
    print(f"  {g!r:28} -> {p.strip()!r}{flag}")

OUT = PHD / "Output_Repro/OCR"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ocr_accuracy.json").write_text(json.dumps({
    "model": str(MODEL),
    "test_dir": str(TEST),
    "crops": len(files),
    "case_sensitive":   {"cer": round(cer, 4),  "wer": round(wer, 4),  "exact": round(ex, 4)},
    "case_insensitive": {"cer": round(cerL, 4), "wer": round(werL, 4), "exact": round(exL, 4)},
    # every prediction, so a disputed figure can be traced to the crop that produced it
    "predictions": [{"gt": g, "pred": p.strip()} for g, p in zip(labels, preds)],
}, indent=2))
print(f"\nwritten -> {OUT / 'ocr_accuracy.json'}")
