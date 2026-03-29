# Detailed Task Execution Guide for EMNLP 2026 Submission

I'll explain each major task with **exact steps, code, and resource requirements** based on your Lipi-Ghor dataset structure and Kaggle/BUET setup.

---

## PHASE 1: SETUP & INFRASTRUCTURE

### Task 1.1: Prepare Test Set from Lipi-Ghor

**What:** Split your 882-hour dataset into train/val/test for benchmarking

**Why:** Need held-out test set to evaluate ASR models fairly (can't use training data for evaluation)

**How:**

```python
# test_split_creator.py
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Load your dataset
ds = load_dataset("Sanjidh090/Lipi-Ghor-bn-882-SSTT")

# Since Lipi-Ghor is in SSTT format, each row has:
# - audio: audio array
# - text: transcription
# - speaker_id: speaker label (if available)
# - duration: clip length
# - video_id/channel: metadata

# Strategy: Stratified sampling by channel/category to preserve domain diversity
df = pd.DataFrame(ds['train'])  # or whatever split name you used

# Calculate hours per channel
channel_stats = df.groupby('channel_name').agg({
    'duration': 'sum',
    'audio': 'count'
}).reset_index()
channel_stats['hours'] = channel_stats['duration'] / 3600

# Sample 5% for test, 5% for validation, 90% for training
# Stratify by channel to maintain domain balance
test_size = 0.05
val_size = 0.05

# Split by video_id (not individual clips) to avoid data leakage
video_ids = df['video_id'].unique()
train_vids, temp_vids = train_test_split(
    video_ids, 
    test_size=(test_size + val_size),
    random_state=42
)
val_vids, test_vids = train_test_split(
    temp_vids,
    test_size=test_size/(test_size + val_size),
    random_state=42
)

# Create splits
train_df = df[df['video_id'].isin(train_vids)]
val_df = df[df['video_id'].isin(val_vids)]
test_df = df[df['video_id'].isin(test_vids)]

print(f"Train: {len(train_df)} clips, {train_df['duration'].sum()/3600:.1f} hours")
print(f"Val: {len(val_df)} clips, {val_df['duration'].sum()/3600:.1f} hours")
print(f"Test: {len(test_df)} clips, {test_df['duration'].sum()/3600:.1f} hours")

# Save splits
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)

# Create test set statistics for paper
test_stats = {
    'total_hours': test_df['duration'].sum() / 3600,
    'num_clips': len(test_df),
    'num_speakers': test_df['speaker_id'].nunique() if 'speaker_id' in test_df else 'N/A',
    'num_videos': test_df['video_id'].nunique(),
    'num_channels': test_df['channel_name'].nunique(),
    'avg_duration': test_df['duration'].mean(),
    'domain_distribution': test_df.groupby('category')['duration'].sum() / test_df['duration'].sum()
}

pd.DataFrame([test_stats]).to_csv('test_set_statistics.csv')
```

**Resources:** Run on your local machine or Kaggle notebook
**Time:** 30 minutes
**Output:** `train_split.csv`, `val_split.csv`, `test_split.csv`, `test_set_statistics.csv`

---

### Task 1.2: Set Up Evaluation Pipeline

**What:** Create standardized scripts to compute WER/CER/DER

**Why:** Need consistent metric computation across all models

**How:**

```python
# metrics.py
import jiwer
from typing import List, Dict
import editdistance
import pandas as pd

def normalize_bengali_text(text: str) -> str:
    """
    Apply your existing normalization pipeline
    Based on your prior Unicode work (য/ড → য়/ড়)
    """
    import re
    from unicode_normalize import normalize  # your existing function
    
    # Apply your normalization rules
    text = normalize(text)  # NFC normalization
    text = text.strip()
    # Add your য/ড conversion logic here
    return text

def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Word Error Rate"""
    # Normalize both
    refs = [normalize_bengali_text(r) for r in references]
    hyps = [normalize_bengali_text(h) for h in hypotheses]
    
    wer = jiwer.wer(refs, hyps)
    return wer * 100  # as percentage

def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """Character Error Rate"""
    refs = [normalize_bengali_text(r) for r in references]
    hyps = [normalize_bengali_text(h) for h in hypotheses]
    
    cer = jiwer.cer(refs, hyps)
    return cer * 100

def compute_detailed_metrics(references: List[str], hypotheses: List[str]) -> Dict:
    """Get insertion, deletion, substitution counts"""
    refs = [normalize_bengali_text(r) for r in references]
    hyps = [normalize_bengali_text(h) for h in hypotheses]
    
    measures = jiwer.compute_measures(refs, hyps)
    
    return {
        'wer': measures['wer'] * 100,
        'cer': jiwer.cer(refs, hyps) * 100,
        'insertions': measures['insertions'],
        'deletions': measures['deletions'],
        'substitutions': measures['substitutions'],
        'hits': measures['hits']
    }

def evaluate_predictions(predictions_file: str, ground_truth_file: str) -> pd.DataFrame:
    """
    Main evaluation function
    predictions_file: CSV with columns [clip_id, predicted_text]
    ground_truth_file: CSV with columns [clip_id, reference_text]
    """
    preds = pd.read_csv(predictions_file)
    refs = pd.read_csv(ground_truth_file)
    
    merged = preds.merge(refs, on='clip_id')
    
    metrics = compute_detailed_metrics(
        merged['reference_text'].tolist(),
        merged['predicted_text'].tolist()
    )
    
    return pd.DataFrame([metrics])

# Install dependencies
# pip install jiwer editdistance
```

**Resources:** Any Python environment
**Time:** 1 hour to write + test
**Output:** `metrics.py` reusable module

---

## PHASE 2: ASR BENCHMARKING

### Task 2.1: Whisper Large-v3 Inference

**What:** Run OpenAI Whisper on your test set

**Why:** Whisper is SOTA baseline - everyone benchmarks against it

**How:**

```python
# whisper_inference.py
import whisper
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_dataset

# Load model (will download ~3GB first time)
model = whisper.load_model("large-v3")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load test set
test_df = pd.read_csv('test_split.csv')

# If you stored audio in HuggingFace dataset, load from there
# Otherwise, load from local files
ds = load_dataset("Sanjidh090/Lipi-Ghor-bn-882-SSTT", split="train")
# Filter to test clip IDs
test_clip_ids = set(test_df['clip_id'])  # assuming you have clip_id column

results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # Get audio array from dataset
    # Assuming you can access by clip_id or index
    audio_array = ds[idx]['audio']['array']  # adjust based on your structure
    
    # Whisper expects 16kHz, your data should already be 16kHz
    result = model.transcribe(
        audio_array,
        language='bn',  # Bengali
        fp16=torch.cuda.is_available(),
        verbose=False
    )
    
    results.append({
        'clip_id': row['clip_id'],
        'predicted_text': result['text'],
        'reference_text': row['text']
    })
    
    # Save intermediate results every 100 clips
    if len(results) % 100 == 0:
        pd.DataFrame(results).to_csv('whisper_v3_predictions_partial.csv', index=False)

# Save final results
predictions_df = pd.DataFrame(results)
predictions_df.to_csv('whisper_v3_predictions.csv', index=False)

# Compute metrics
from metrics import evaluate_predictions
metrics = evaluate_predictions(
    'whisper_v3_predictions.csv',
    'test_split.csv'  # has reference_text column
)
metrics.to_csv('whisper_v3_metrics.csv', index=False)
print(f"Whisper-v3 WER: {metrics['wer'].values[0]:.2f}%")
```

**Resources:**
- **GPU:** Kaggle T4 (free) or P100 (30 hrs/week free)
- **Time:** ~0.3 sec/clip → 40 hours test set = ~3-4 hours inference
- **Memory:** ~6GB VRAM for large-v3

**Kaggle Setup:**
```python
# In Kaggle notebook
# Settings: GPU T4 x2 accelerator
# Persistence: Enable Internet (for model download)

!pip install openai-whisper

# Your inference script here
```

**Expected Output:**
```
whisper_v3_predictions.csv:
clip_id,predicted_text,reference_text
clip_001,আমি ভালো আছি,আমি ভালো আছি
clip_002,বাংলাদেশ একটি সুন্দর দেশ,বাংলাদেশ একটি সুন্দর দেশ
...

whisper_v3_metrics.csv:
wer,cer,insertions,deletions,substitutions,hits
12.5,6.3,150,200,350,5000
```

---

### Task 2.2: Meta MMS 1B Inference

**What:** Run Meta's Massively Multilingual Speech model

**Why:** MMS trained on 1000+ languages, good multilingual baseline

**How:**

```python
# mms_inference.py
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

# Load MMS model for Bengali
model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Set target language to Bengali
processor.tokenizer.set_target_lang("ben")  # ISO 639-3 code for Bengali
model.load_adapter("ben")

test_df = pd.read_csv('test_split.csv')

results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    # Load audio (same as before)
    audio_array = ds[idx]['audio']['array']
    
    # Preprocess
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # Decode
    predicted_ids = torch.argmax(outputs, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    results.append({
        'clip_id': row['clip_id'],
        'predicted_text': transcription,
        'reference_text': row['text']
    })
    
    if len(results) % 100 == 0:
        pd.DataFrame(results).to_csv('mms_predictions_partial.csv', index=False)

pd.DataFrame(results).to_csv('mms_predictions.csv', index=False)

# Evaluate
from metrics import evaluate_predictions
metrics = evaluate_predictions('mms_predictions.csv', 'test_split.csv')
metrics.to_csv('mms_metrics.csv', index=False)
```

**Resources:**
- GPU: Same as Whisper (T4/P100)
- Time: ~0.2 sec/clip → 2-3 hours
- Memory: ~8GB VRAM

**Potential Issue:** MMS Bengali adapter might not exist or perform poorly
**Fix:** Check available adapters first:
```python
from transformers import Wav2Vec2ForCTC
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
print(model.config.adapter_attn_dim)  # check if 'ben' is available
```

---

### Task 2.3: SeamlessM4T v2 Inference

**What:** Run Meta's speech-to-text model

**Why:** Strong multilingual baseline with recent architecture

**How:**

```python
# seamless_inference.py
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
import torch
import pandas as pd
from tqdm import tqdm

model_id = "facebook/seamless-m4t-v2-large"
processor = AutoProcessor.from_pretrained(model_id)
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

test_df = pd.read_csv('test_split.csv')
results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    audio_array = ds[idx]['audio']['array']
    
    # Process audio
    audio_inputs = processor(
        audios=audio_array,
        return_tensors="pt",
        sampling_rate=16000
    ).to(device)
    
    # Generate transcription
    # tgt_lang for Bengali: "ben" 
    output_tokens = model.generate(
        **audio_inputs,
        tgt_lang="ben",
        generate_speech=False
    )
    
    # Decode
    transcription = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    
    results.append({
        'clip_id': row['clip_id'],
        'predicted_text': transcription,
        'reference_text': row['text']
    })
    
    if len(results) % 100 == 0:
        pd.DataFrame(results).to_csv('seamless_predictions_partial.csv', index=False)

pd.DataFrame(results).to_csv('seamless_predictions.csv', index=False)

from metrics import evaluate_predictions
metrics = evaluate_predictions('seamless_predictions.csv', 'test_split.csv')
metrics.to_csv('seamless_metrics.csv', index=False)
```

**Resources:** Similar to previous models
**Time:** 3-4 hours

---

### Task 2.4: IndicWhisper Inference

**What:** Run AI4Bharat's Indic-language fine-tuned Whisper

**Why:** Specifically tuned for Indian languages, should perform well on Bengali

**How:**

```python
# indicwhisper_inference.py
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import pandas as pd
from tqdm import tqdm

# AI4Bharat IndicWhisper
model_id = "AI4Bharat/indicwhisper-large"  # check exact model name on HuggingFace
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

test_df = pd.read_csv('test_split.csv')
results = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    audio_array = ds[idx]['audio']['array']
    
    # Process
    input_features = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    # Force Bengali language token
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="bengali", task="transcribe")
    
    # Generate
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids
    )
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    results.append({
        'clip_id': row['clip_id'],
        'predicted_text': transcription,
        'reference_text': row['text']
    })

pd.DataFrame(results).to_csv('indicwhisper_predictions.csv', index=False)

from metrics import evaluate_predictions
metrics = evaluate_predictions('indicwhisper_predictions.csv', 'test_split.csv')
metrics.to_csv('indicwhisper_metrics.csv', index=False)
```

---

## PHASE 3: DIARIZATION BENCHMARKING

### Task 3.1: Pyannote DER Evaluation

**What:** Measure how well pyannote identifies speakers vs your ground truth

**Why:** You used pyannote to create the dataset - need to show it worked well

**How:**

```python
# pyannote_evaluation.py
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
import pandas as pd
from pyannote.core import Annotation, Segment
import torchaudio

# Load pyannote pipeline (you already have this from dataset creation)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"  # from your previous work
)

# Load ground truth
# Assuming you have RTTM files or JSON with speaker timestamps
def load_ground_truth_rttm(rttm_path):
    """Load ground truth from RTTM format"""
    annotation = Annotation()
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation

# If your ground truth is in JSON (from diarization_transcription_final/)
def load_ground_truth_json(json_path):
    import json
    annotation = Annotation()
    with open(json_path) as f:
        data = json.load(f)
    for segment in data:
        start = segment['start']
        end = segment['end']
        speaker = segment['speaker']
        annotation[Segment(start, end)] = speaker
    return annotation

# Evaluate on test set
metric = DiarizationErrorRate()

test_files = []  # list of your test audio files
for audio_file in test_files:
    # Run pyannote inference
    diarization = pipeline(audio_file)
    
    # Load ground truth
    ground_truth = load_ground_truth_json(audio_file.replace('.wav', '.json'))
    
    # Compute DER
    metric(ground_truth, diarization)

# Get overall DER
der_components = metric.report(display=False)
print(f"DER: {der_components['diarization error rate']:.2%}")
print(f"False Alarm: {der_components['false alarm']:.2%}")
print(f"Missed Detection: {der_components['missed detection']:.2%}")
print(f"Confusion: {der_components['confusion']:.2%}")

# Save results
results = {
    'system': 'pyannote-3.1',
    'DER': der_components['diarization error rate'] * 100,
    'FA': der_components['false alarm'] * 100,
    'Miss': der_components['missed detection'] * 100,
    'Confusion': der_components['confusion'] * 100
}
pd.DataFrame([results]).to_csv('pyannote_der_results.csv', index=False)
```

**Problem:** You might not have ground truth speaker labels
**Solution:** 
1. Manually annotate 50-100 conversations (subset of test set)
2. Use majority voting from multiple diarization runs
3. Or skip this and only report ASR metrics (acceptable for resource paper)

**Resources:** 
- GPU: T4 sufficient
- Time: 30 min - 2 hours depending on test set size
- Manual annotation time: 10-20 hours if needed

---

### Task 3.2: NeMo MSDD Diarization

**What:** Run NVIDIA's Multi-Scale Diarization Decoder

**Why:** SOTA diarization system to compare against pyannote

**How:**

```python
# nemo_msdd_inference.py
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import pandas as pd

# Download pretrained MSDD model
msdd_model = nemo_asr.models.ClusteringDiarizer.from_pretrained("nvidia/speakerverification_en_titanet_large")

# Or use MSDD directly
# This is complex - NeMo requires config files

# Simplified approach: use NeMo's diarization script
!python /path/to/NeMo/examples/speaker_tasks/diarization/offline_diarization.py \
    --config-path=/path/to/config \
    --config-name=diar_infer_telephonic.yaml \
    diarizer.manifest_filepath=test_manifest.json \
    diarizer.out_dir=outputs/

# Then evaluate same as pyannote
```

**Reality Check:** NeMo MSDD is complex to set up properly
**Recommendation:** 
- If you get it working in 2-3 hours, great
- If not, skip it and only use pyannote
- Paper can say "we evaluated with pyannote-3.1, the pipeline used for annotation"

---

## PHASE 4: COMPARATIVE FINE-TUNING (MOST IMPORTANT)

### Task 4.1: Download Comparison Datasets

**What:** Get Mozilla Common Voice & FLEURS Bengali data

**Why:** Need to show Lipi-Ghor is better than existing datasets

**How:**

```python
# download_comparison_datasets.py
from datasets import load_dataset

# Mozilla Common Voice Bengali
cv_bn = load_dataset("mozilla-foundation/common_voice_11_0", "bn", use_auth_token=True)
# You need to accept terms on HuggingFace first

# Check size
cv_train = cv_bn['train']
cv_test = cv_bn['test']
print(f"Common Voice Bengali: {len(cv_train)} train, {len(cv_test)} test")
# Usually ~30-50 hours total

# Google FLEURS Bengali
fleurs_bn = load_dataset("google/fleurs", "bn_in")  # bn_in = Bengali India
print(f"FLEURS: {len(fleurs_bn['train'])} train, {len(fleurs_bn['test'])} test")
# Usually ~10 hours

# Save locally for faster access
cv_bn.save_to_disk("data/common_voice_bn")
fleurs_bn.save_to_disk("data/fleurs_bn")
```

**Resources:** 10 GB disk space
**Time:** 30 min download

---

### Task 4.2: Fine-tune Whisper on Each Dataset

**What:** Train Whisper-small on (1) Common Voice, (2) FLEURS, (3) Lipi-Ghor

**Why:** THIS IS THE KEY RESULT - shows Lipi-Ghor improves ASR performance

**How (you already know this from your prior work):**

```python
# finetune_whisper.py
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_from_disk
import torch

model_id = "openai/whisper-small"  # Use small for faster training
processor = WhisperProcessor.from_pretrained(model_id, language="bengali", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# Prepare dataset
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Load training dataset (Common Voice as example)
train_dataset = load_from_disk("data/common_voice_bn")['train']
train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-bn-commonvoice",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,  # CRITICAL: you learned this from your L40S collapse
    num_train_epochs=3,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # WER computation
    tokenizer=processor.feature_extractor,
)

# Train
trainer.train()

# Save
trainer.save_model("whisper-small-bn-commonvoice-final")
```

**Repeat for:**
1. Common Voice Bengali → `whisper-small-bn-cv/`
2. FLEURS Bengali → `whisper-small-bn-fleurs/`
3. Lipi-Ghor → `whisper-small-bn-lipighor/`

**Resources:**
- GPU: Kaggle P100 or Colab A100
- Time per training run: 
  - Common Voice (~40 hours): 6-8 hours training
  - FLEURS (~10 hours): 2-3 hours
  - Lipi-Ghor (~800 hours): 48-72 hours
- **CRITICAL:** Use same number of epochs/steps for fair comparison

**Comparison Strategy:**
Since Lipi-Ghor is 20x larger, need fair comparison:

**Option A:** Train all for same wall-clock time
- Train each for 10 hours
- Report WER on common test set

**Option B:** Train all for same number of steps
- Calculate steps_per_epoch for each
- Train all for 10,000 steps
- Report WER

**Option C:** Train until convergence (best for paper)
- Use early stopping
- Train each until validation WER stops improving
- Report final WER + training hours in table

**I recommend Option C** - shows real performance potential

---

### Task 4.3: Evaluate All Fine-tuned Models

**What:** Run all 3 fine-tuned models + baseline on same test set

**Why:** Create the comparison table for the paper

**How:**

```python
# evaluate_all_models.py
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import pandas as pd
from tqdm import tqdm

models = {
    'Baseline (no fine-tune)': 'openai/whisper-small',
    'Fine-tuned on Common Voice': './whisper-small-bn-cv',
    'Fine-tuned on FLEURS': './whisper-small-bn-fleurs',
    'Fine-tuned on Lipi-Ghor': './whisper-small-bn-lipighor'
}

# Load YOUR held-out test set (from Lipi-Ghor)
test_df = pd.read_csv('test_split.csv')

all_results = []

for model_name, model_path in models.items():
    print(f"\nEvaluating {model_name}...")
    
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to('cuda')
    
    predictions = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_array = ds[idx]['audio']['array']
        
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to('cuda')
        
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        predictions.append(transcription)
    
    # Compute WER
    from metrics import compute_wer, compute_cer
    wer = compute_wer(test_df['text'].tolist(), predictions)
    cer = compute_cer(test_df['text'].tolist(), predictions)
    
    all_results.append({
        'Model': model_name,
        'WER (%)': f"{wer:.2f}",
        'CER (%)': f"{cer:.2f}",
        'Training Data': model_path.split('-')[-1] if 'fine' in model_name else 'None',
        'Training Hours': get_training_hours(model_path)  # you track this
    })

# Create comparison table
results_df = pd.DataFrame(all_results)
results_df.to_csv('comparative_finetuning_results.csv', index=False)
print("\n" + results_df.to_string(index=False))
```

**Expected Output (example):**
```
Model                          | WER (%) | CER (%) | Training Data | Training Hours
-------------------------------|---------|---------|---------------|---------------
Baseline (no fine-tune)        | 28.5    | 14.2    | None          | 0
Fine-tuned on Common Voice     | 22.3    | 11.8    | Common Voice  | 45
Fine-tuned on FLEURS           | 25.1    | 13.4    | FLEURS        | 10
Fine-tuned on Lipi-Ghor        | 16.7    | 8.9     | Lipi-Ghor     | 882
```

**This table goes directly in your paper** - shows Lipi-Ghor's value

---

## PHASE 5: HUMAN EVALUATION

### Task 5.1: Create Annotation Interface

**What:** Build simple UI for humans to rate transcription quality

**Why:** Paper needs human eval to validate WER scores

**How:**

```html
<!-- annotation_interface.html -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Bengali ASR Annotation</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
        .sample { border: 1px solid #ccc; padding: 20px; margin: 20px 0; }
        audio { width: 100%; margin: 10px 0; }
        textarea { width: 100%; height: 80px; font-size: 16px; }
        .rating { margin: 10px 0; }
        button { background: #4CAF50; color: white; padding: 10px 20px; font-size: 16px; }
    </style>
</head>
<body>
    <h1>Bengali ASR Quality Annotation</h1>
    <p>নির্দেশনা: অডিও শুনুন, রেফারেন্স পড়ুন, তারপর মডেল আউটপুট রেট করুন (1=খুব খারাপ, 5=নিখুঁত)</p>
    
    <div id="samples"></div>
    
    <script>
        // Load samples from JSON
        let samples = []; // loaded from samples.json
        let currentIdx = 0;
        let annotations = [];
        
        function loadSample(idx) {
            const sample = samples[idx];
            const container = document.getElementById('samples');
            container.innerHTML = `
                <div class="sample">
                    <h3>Sample ${idx + 1} of ${samples.length}</h3>
                    <audio controls src="${sample.audio_url}"></audio>
                    
                    <p><strong>Reference (Ground Truth):</strong><br>
                    <textarea readonly>${sample.reference}</textarea></p>
                    
                    <p><strong>Model Output:</strong><br>
                    <textarea readonly>${sample.prediction}</textarea></p>
                    
                    <div class="rating">
                        <strong>Quality Rating:</strong><br>
                        <label><input type="radio" name="rating" value="1"> 1 - Very Poor</label>
                        <label><input type="radio" name="rating" value="2"> 2 - Poor</label>
                        <label><input type="radio" name="rating" value="3"> 3 - Acceptable</label>
                        <label><input type="radio" name="rating" value="4"> 4 - Good</label>
                        <label><input type="radio" name="rating" value="5"> 5 - Perfect</label>
                    </div>
                    
                    <button onclick="submitRating()">Submit & Next</button>
                </div>
            `;
        }
        
        function submitRating() {
            const rating = document.querySelector('input[name="rating"]:checked');
            if (!rating) { alert('Please select a rating'); return; }
            
            annotations.push({
                sample_id: samples[currentIdx].id,
                rating: parseInt(rating.value),
                annotator: prompt('Your name/ID:')
            });
            
            currentIdx++;
            if (currentIdx < samples.length) {
                loadSample(currentIdx);
            } else {
                // Save annotations
                const blob = new Blob([JSON.stringify(annotations, null, 2)], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'annotations.json';
                a.click();
                alert('All done! Thank you!');
            }
        }
        
        // Load samples on page load
        fetch('samples.json').then(r => r.json()).then(data => {
            samples = data;
            loadSample(0);
        });
    </script>
</body>
</html>
```

**Prepare samples:**
```python
# prepare_human_eval_samples.py
import pandas as pd
import random
import json

# Sample 100 clips from test set
test_df = pd.read_csv('test_split.csv')
sampled = test_df.sample(100, random_state=42)

# Get predictions from best model
preds = pd.read_csv('whisper_v3_predictions.csv')
merged = sampled.merge(preds, on='clip_id')

# Create samples JSON
samples = []
for idx, row in merged.iterrows():
    samples.append({
        'id': row['clip_id'],
        'audio_url': f"audio/{row['clip_id']}.wav",  # upload to Google Drive or host locally
        'reference': row['reference_text'],
        'prediction': row['predicted_text']
    })

with open('samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
```

**Recruit annotators:**
- Ask 2-3 teammates (Risalat, Fuad, Bayazid)
- Ask 1-2 Bengali speakers from BUET CSE dept
- Pay 500 BDT per person (50 samples each)

**Time:** 3-5 minutes per sample → 5-8 hours total per annotator

---

### Task 5.2: Compute Inter-Annotator Agreement

**What:** Measure how much annotators agree

**Why:** Shows annotation quality is reliable

**How:**

```python
# compute_kappa.py
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np

# Load annotations from all annotators
annotator1 = pd.read_json('annotations_person1.json')
annotator2 = pd.read_json('annotations_person2.json')
annotator3 = pd.read_json('annotations_person3.json')

# Merge on sample_id
merged = annotator1.merge(annotator2, on='sample_id', suffixes=('_1', '_2'))
merged = merged.merge(annotator3, on='sample_id')
merged = merged.rename(columns={'rating': 'rating_3'})

# Cohen's Kappa for each pair
kappa_12 = cohen_kappa_score(merged['rating_1'], merged['rating_2'])
kappa_13 = cohen_kappa_score(merged['rating_1'], merged['rating_3'])
kappa_23 = cohen_kappa_score(merged['rating_2'], merged['rating_3'])

print(f"Kappa (Annotator 1-2): {kappa_12:.3f}")
print(f"Kappa (Annotator 1-3): {kappa_13:.3f}")
print(f"Kappa (Annotator 2-3): {kappa_23:.3f}")
print(f"Average Kappa: {np.mean([kappa_12, kappa_13, kappa_23]):.3f}")

# Fleiss' Kappa for 3+ annotators
from statsmodels.stats.inter_rater import fleiss_kappa

# Create rating matrix (rows=samples, cols=annotators)
ratings = merged[['rating_1', 'rating_2', 'rating_3']].values
fleiss = fleiss_kappa(ratings, method='fleiss')
print(f"Fleiss' Kappa: {fleiss:.3f}")

# Interpretation:
# < 0.20 = Poor
# 0.21-0.40 = Fair  
# 0.41-0.60 = Moderate
# 0.61-0.80 = Substantial
# 0.81-1.00 = Almost Perfect
```

**Expected:** κ = 0.6-0.8 (substantial agreement)

---

## TIMELINE REALITY CHECK

**Week 1 (Apr 1-5):**
- Set up test split ✓
- Download all models ✓
- Run Whisper-v3 inference ✓

**Week 2 (Apr 6-12):**
- Run MMS, SeamlessM4T, IndicWhisper ✓
- Start fine-tuning on Common Voice ✓

**Week 3 (Apr 13-19):**
- Finish fine-tuning experiments
- Run diarization evaluation (if time)

**Week 4-5 (Apr 20-May 3):**
- Human evaluation
- Download FLEURS, Common Voice
- Final comparative evaluation

**Week 6 (May 4-10):**
- Compile all results
- Create tables/figures
- Start paper draft

**Week 7-8 (May 11-25):**
- Write paper with Nishat
- Submit to ARR May 25

---

## CRITICAL SUCCESS FACTORS

**What you MUST complete:**
1. ✅ Whisper-v3 baseline on test set
2. ✅ Fine-tune on Lipi-Ghor vs Common Voice comparison
3. ✅ Comparative WER table

**What's important but droppable if time runs out:**
4. MMS/SeamlessM4T/IndicWhisper (nice to have, but not critical)
5. Diarization DER (you can cite your dataset creation pipeline)
6. Human evaluation (can be minimal - 50 samples, 2 annotators)

**Parallelization strategy:**
- **Risalat:** Whisper inference + Common Voice download
- **Fuad:** MMS/SeamlessM4T setup + FLEURS download
- **Bayazid:** Fine-tuning experiments
- **Sanjid:** Metrics pipeline + paper draft coordination

Let me know which specific task you want MORE detail on - I can provide complete runnable code for any of these.
