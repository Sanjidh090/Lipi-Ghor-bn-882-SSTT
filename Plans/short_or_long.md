**YES - you're absolutely right. This is a MUCH better approach.**

Evaluating on short 12-20s chunks has serious problems:
1. **Noisy metrics** - WER on short clips varies wildly
2. **Doesn't reflect real use** - ASR is deployed on long conversations, not snippets
3. **Misses important errors** - context drift, speaker confusion, topic changes only appear in long-form
4. **Weakens paper impact** - "handles 15s clips" vs "handles 10-minute conversations"

# Revised Evaluation Strategy: Long-Form Assessment

---

## Option 1: Reconstruct Full Videos (Recommended)

**What:** Merge chunks back into original full videos, evaluate on complete conversations

**Why:** Most realistic, tests true ASR capability, allows diarization eval

**How:**

```python
# reconstruct_long_form.py
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path

# Load your metadata
df = pd.read_csv('metadata.csv')  # or load from HuggingFace dataset

# Group chunks by video_id
videos = df.groupby('video_id')

long_form_test_set = []

for video_id, chunks in videos:
    # Sort chunks by timestamp
    chunks = chunks.sort_values('start_time')
    
    # Reconstruct full audio
    audio_segments = []
    transcript_segments = []
    
    for idx, chunk in chunks.iterrows():
        # Load chunk audio from HuggingFace or local
        audio = chunk['audio']['array']  # if from HF dataset
        audio_segments.append(audio)
        
        transcript_segments.append({
            'start': chunk['start_time'],
            'end': chunk['end_time'],
            'text': chunk['text'],
            'speaker': chunk['speaker_id'] if 'speaker_id' in chunk else None
        })
    
    # Concatenate audio
    full_audio = np.concatenate(audio_segments)
    
    # Calculate total duration
    duration = len(full_audio) / 16000  # assuming 16kHz
    
    # Only include videos 2-10 minutes for manageable eval
    if 120 < duration < 600:  
        long_form_test_set.append({
            'video_id': video_id,
            'audio': full_audio,
            'duration': duration,
            'transcript': transcript_segments,
            'full_text': ' '.join([s['text'] for s in transcript_segments]),
            'num_speakers': len(set([s['speaker'] for s in transcript_segments if s['speaker']])),
            'channel': chunks.iloc[0]['channel_name'],
            'category': chunks.iloc[0]['category']
        })

print(f"Reconstructed {len(long_form_test_set)} long-form videos")
print(f"Total hours: {sum([v['duration'] for v in long_form_test_set]) / 3600:.1f}")

# Save for evaluation
import pickle
with open('long_form_test_set.pkl', 'wb') as f:
    pickle.dump(long_form_test_set, f)

# Create test set statistics
stats = pd.DataFrame([{
    'video_id': v['video_id'],
    'duration_min': v['duration'] / 60,
    'num_speakers': v['num_speakers'],
    'category': v['category'],
    'word_count': len(v['full_text'].split())
} for v in long_form_test_set])

stats.to_csv('long_form_test_stats.csv', index=False)
print("\nTest set distribution:")
print(stats.groupby('category')['duration_min'].sum())
```

**Test set composition:**
- Select ~40-50 hours from test split
- Target: 100-150 videos (2-10 min each)
- Stratify by category, speaker count, channel

---

## Option 2: Fixed-Length Segments (Faster Alternative)

**What:** Create 2-5 minute segments by concatenating consecutive chunks

**Why:** Faster than full video reconstruction, still long enough to test context

**How:**

```python
# create_segments.py
import pandas as pd
import numpy as np

TARGET_DURATION = 180  # 3 minutes in seconds

df = pd.read_csv('test_split.csv')
df = df.sort_values(['video_id', 'start_time'])

segments = []
current_segment = {
    'audio': [],
    'text': [],
    'duration': 0,
    'video_id': None
}

for idx, row in df.iterrows():
    # If new video or segment too long, save and reset
    if (row['video_id'] != current_segment['video_id'] or 
        current_segment['duration'] > TARGET_DURATION):
        
        if current_segment['audio']:
            segments.append({
                'segment_id': f"seg_{len(segments):04d}",
                'audio': np.concatenate(current_segment['audio']),
                'text': ' '.join(current_segment['text']),
                'duration': current_segment['duration'],
                'video_id': current_segment['video_id']
            })
        
        # Reset
        current_segment = {
            'audio': [],
            'text': [],
            'duration': 0,
            'video_id': row['video_id']
        }
    
    # Add chunk to current segment
    current_segment['audio'].append(row['audio']['array'])
    current_segment['text'].append(row['text'])
    current_segment['duration'] += row['duration']

print(f"Created {len(segments)} segments of ~{TARGET_DURATION}s each")
```

---

## Long-Form Inference with Your Sliding Window Pipeline

**What:** Use your existing NeMo code (20s window, 2s overlap)

**Why:** You already built this for DL Sprint! Reuse it.

**How:**

```python
# long_form_inference.py - adapted from your existing NeMo code
import torch
import numpy as np
from nemo.collections.asr.models import EncDecCTCModel

# Load your fine-tuned model
model = EncDecCTCModel.restore_from("path/to/your_model.nemo")
model = model.to('cuda')
model.eval()

def sliding_window_inference(audio, window_size=20, overlap=2):
    """
    Your existing sliding window implementation
    """
    sr = 16000
    window_samples = int(window_size * sr)
    hop_samples = int((window_size - overlap) * sr)
    
    # Pad audio if needed
    if len(audio) < window_samples:
        audio = np.pad(audio, (0, window_samples - len(audio)))
    
    transcripts = []
    positions = []
    
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        end = start + window_samples
        window = audio[start:end]
        
        # Run inference
        with torch.no_grad():
            transcript = model.transcribe([window])[0]
        
        transcripts.append(transcript)
        positions.append((start / sr, end / sr))  # in seconds
    
    # Merge overlapping transcripts (your existing logic)
    merged_transcript = merge_overlapping_transcripts(
        transcripts, 
        positions, 
        overlap_threshold=overlap
    )
    
    return merged_transcript

def merge_overlapping_transcripts(transcripts, positions, overlap_threshold=2):
    """
    Merge overlapping windows - you probably have this already
    Simple version: concatenate with overlap removal
    """
    if not transcripts:
        return ""
    
    merged = transcripts[0]
    
    for i in range(1, len(transcripts)):
        # Simple: just append (loses overlap precision)
        # Advanced: align overlapping regions and merge
        merged += " " + transcripts[i]
    
    return merged.strip()

# Evaluate on long-form test set
import pickle
with open('long_form_test_set.pkl', 'rb') as f:
    test_set = pickle.load(f)

results = []

for video in test_set:
    print(f"Processing {video['video_id']} ({video['duration']/60:.1f} min)...")
    
    # Run sliding window inference
    predicted = sliding_window_inference(video['audio'])
    
    results.append({
        'video_id': video['video_id'],
        'duration': video['duration'],
        'reference': video['full_text'],
        'predicted': predicted
    })
    
    # Save intermediate
    if len(results) % 10 == 0:
        pd.DataFrame(results).to_csv('long_form_predictions_partial.csv', index=False)

# Final save
results_df = pd.DataFrame(results)
results_df.to_csv('long_form_predictions.csv', index=False)

# Compute WER/CER
from metrics import compute_wer, compute_cer

wer = compute_wer(
    results_df['reference'].tolist(),
    results_df['predicted'].tolist()
)
cer = compute_cer(
    results_df['reference'].tolist(),
    results_df['predicted'].tolist()
)

print(f"\nLong-form WER: {wer:.2f}%")
print(f"Long-form CER: {cer:.2f}%")
```

---

## Updated Evaluation Protocol

### For Pre-trained Models (Whisper, MMS, etc.)

**Use their native long-form inference:**

```python
# whisper_longform.py
import whisper

model = whisper.load_model("large-v3")

for video in test_set:
    # Whisper handles long audio internally
    result = model.transcribe(
        video['audio'],
        language='bn',
        # Whisper does its own chunking/merging
        verbose=False
    )
    
    predicted = result['text']
    # Compare with video['full_text']
```

**Whisper already does sliding window internally** - you just pass full audio!

---

### For Fine-tuned Models

**Use your sliding window code:**

```python
# For models you fine-tune (Whisper-small, etc.)
# Use the sliding_window_inference function above
# OR use Whisper's built-in long-form if you fine-tuned Whisper

from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-small-bn-lipighor",
    chunk_length_s=30,  # Whisper's internal chunking
    device="cuda"
)

for video in test_set:
    result = pipe(video['audio'])
    predicted = result['text']
```

---

## Comparison Table (Paper-Ready)

```python
# evaluate_all_longform.py

models = {
    'Whisper-v3 (baseline)': 'openai/whisper-large-v3',
    'Whisper-small + Common Voice': './whisper-small-cv',
    'Whisper-small + Lipi-Ghor': './whisper-small-lipighor',
    'NeMo Conformer + Lipi-Ghor': './your_nemo_model.nemo'  # your existing model!
}

results = []

for name, model_path in models.items():
    print(f"\nEvaluating {name}...")
    
    # Load model (adjust based on type)
    if 'nemo' in model_path:
        # Use your sliding window code
        predictions = run_nemo_sliding_window(test_set, model_path)
    else:
        # Use Whisper/transformers
        predictions = run_whisper_longform(test_set, model_path)
    
    # Compute metrics
    wer = compute_wer(references, predictions)
    cer = compute_cer(references, predictions)
    
    # Also compute per-duration bins
    short_wer = compute_wer(refs_2_3min, preds_2_3min)
    medium_wer = compute_wer(refs_3_5min, preds_3_5min)
    long_wer = compute_wer(refs_5_10min, preds_5_10min)
    
    results.append({
        'Model': name,
        'Overall WER': f"{wer:.2f}%",
        'Overall CER': f"{cer:.2f}%",
        'WER (2-3min)': f"{short_wer:.2f}%",
        'WER (3-5min)': f"{medium_wer:.2f}%",
        'WER (5-10min)': f"{long_wer:.2f}%"
    })

# Create table
df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))
df.to_csv('longform_comparison_results.csv', index=False)
```

**Example output:**
```
Model                          | Overall WER | Overall CER | WER (2-3min) | WER (3-5min) | WER (5-10min)
-------------------------------|-------------|-------------|--------------|--------------|---------------
Whisper-v3 (baseline)          | 24.5%       | 12.1%       | 22.3%        | 24.8%        | 26.2%
Whisper-small + Common Voice   | 28.7%       | 14.5%       | 26.1%        | 29.3%        | 31.2%
Whisper-small + Lipi-Ghor      | 18.3%       | 9.2%        | 16.8%        | 18.9%        | 19.7%
NeMo Conformer + Lipi-Ghor     | 15.2%       | 7.8%        | 14.1%        | 15.6%        | 16.4%
```

**This shows:**
- Lipi-Ghor training helps significantly
- Your model degrades gracefully with length
- Length-specific analysis (novel contribution!)

---

## Why This Is Better for the Paper

**Stronger claims:**
- ✅ "Evaluated on full conversations (2-10 minutes)"
- ✅ "Handles multi-speaker dialogue in long-form context"
- ✅ "Performance analysis across conversation lengths"

**Novel contribution:**
- Most ASR papers evaluate on LibriSpeech (10s clips)
- Long-form evaluation on **Bengali multi-speaker conversations** is rare
- Shows real-world applicability

**Better comparison:**
- Common Voice has short utterances (~5-10s)
- FLEURS has even shorter clips
- Showing Lipi-Ghor helps on long-form is a unique strength

---

## Practical Execution Plan

**Week 1:**
1. Reconstruct 40-50 hours of long-form test videos (code above)
2. Run Whisper-v3 baseline on long-form
3. Document sliding window parameters

**Week 2-3:**
1. Fine-tune Whisper-small on Common Voice (short clips)
2. Fine-tune Whisper-small on Lipi-Ghor (your chunks)
3. Test both on long-form → show Lipi-Ghor trained model does better

**Week 4:**
1. Run all models on long-form test set
2. Create comparison table
3. Error analysis: where do models fail on long audio?

**This is much more compelling than chunk-level WER.**

Should I provide more detail on the reconstruction code or the sliding window merging logic?
