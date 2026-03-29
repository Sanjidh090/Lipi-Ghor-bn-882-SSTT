# EMNLP 2026 Submission Plan for Lipi-Ghor
**Target Deadline:** May 25, 2026 (ARR May cycle)  
**Time Available:** ~8 weeks from today

---

## Phase 1: Immediate Setup (Week 1 - April 5)

### Task 1.1: Evaluation Infrastructure Setup
**Priority: CRITICAL - Start Monday**

- [ ] Set up dedicated Kaggle/Colab environment for benchmarking
- [ ] Create evaluation pipeline repository structure
  - [ ] `/benchmarks/asr/` - ASR model evaluations
  - [ ] `/benchmarks/diarization/` - Diarization evaluations
  - [ ] `/benchmarks/comparative/` - Cross-corpus comparisons
  - [ ] `/results/` - WER/CER/DER outputs
- [ ] Prepare held-out test set from Lipi-Ghor
  - [ ] Sample 5-10% stratified by speaker, duration, topic
  - [ ] Document test set statistics (hours, speakers, domains)
  - [ ] Create separate validation split if needed
- [ ] Set up metrics computation scripts
  - [ ] WER calculation (jiwer library)
  - [ ] CER calculation
  - [ ] DER calculation (pyannote.metrics)

### Task 1.2: Model Downloads & Environment Setup
**Owner: Divide across team**

- [ ] **Whisper large-v3** (Risalat)
  - [ ] Install: `pip install openai-whisper`
  - [ ] Test inference on sample
  - [ ] Document GPU requirements
  
- [ ] **Meta MMS 1B** (Fuad)
  - [ ] Clone from HuggingFace: `facebook/mms-1b-all`
  - [ ] Set up Bengali language code
  - [ ] Test inference pipeline
  
- [ ] **SeamlessM4T v2** (Bayazid)
  - [ ] Install: `pip install seamless-communication`
  - [ ] Configure for Bengali ASR
  - [ ] Test on sample clips
  
- [ ] **IndicWhisper** (Sanjid)
  - [ ] Download from: `AI4Bharat/indicwhisper`
  - [ ] Check Bengali-specific fine-tuning
  - [ ] Verify compatibility with your data format

### Task 1.3: Diarization Tools Setup

- [ ] **pyannote-audio 3.x** (already used, but verify)
  - [ ] Confirm version compatibility
  - [ ] Set up ground truth comparison pipeline
  - [ ] Create DER computation script
  
- [ ] **NeMo MSDD**
  - [ ] Install NVIDIA NeMo: `pip install nemo_toolkit[asr]`
  - [ ] Download pretrained MSDD model
  - [ ] Test on sample conversations
  - [ ] Document setup issues/requirements

---

## Phase 2: ASR Benchmarking (Weeks 2-3 - April 6-19)

### Task 2.1: Run Inference on Test Set
**Parallel execution - assign one model per team member**

For each model, create standardized output:
- [ ] **Whisper large-v3**
  - [ ] Batch inference script
  - [ ] Save predictions: `predictions_whisper_v3.json`
  - [ ] Compute WER, CER
  - [ ] Log inference time, GPU memory
  
- [ ] **MMS 1B**
  - [ ] Batch inference script
  - [ ] Save predictions: `predictions_mms.json`
  - [ ] Compute WER, CER
  - [ ] Note any language-specific issues
  
- [ ] **SeamlessM4T v2**
  - [ ] Batch inference script
  - [ ] Save predictions: `predictions_seamless.json`
  - [ ] Compute WER, CER
  - [ ] Track processing bottlenecks
  
- [ ] **IndicWhisper**
  - [ ] Batch inference script
  - [ ] Save predictions: `predictions_indicwhisper.json`
  - [ ] Compute WER, CER
  - [ ] Compare with base Whisper

### Task 2.2: Error Analysis
- [ ] Create confusion matrix for common errors
- [ ] Identify systematic failure modes (domain-specific, speaker characteristics)
- [ ] Generate example outputs for paper (best/worst cases)
- [ ] Document Unicode normalization issues (based on your prior work)

---

## Phase 3: Diarization Benchmarking (Week 3 - April 13-19)

### Task 3.1: Ground Truth Preparation
- [ ] Select subset with verified speaker labels
- [ ] Create RTTM format ground truth files
- [ ] Document annotation methodology

### Task 3.2: Run Diarization Systems
- [ ] **pyannote 3.x**
  - [ ] Re-run on test set (you already have results, but standardize)
  - [ ] Compute DER against ground truth
  - [ ] Break down: False Alarm, Missed Speech, Speaker Confusion
  
- [ ] **NeMo MSDD**
  - [ ] Run inference on same test set
  - [ ] Compute DER with same metrics
  - [ ] Compare speaker embedding quality

### Task 3.3: Analysis
- [ ] Compare DER across multi-speaker vs 2-speaker conversations
- [ ] Analyze performance by conversation length
- [ ] Document where each system excels/fails

---

## Phase 4: Comparative Evaluation (Weeks 4-5 - April 20-May 3)

### Task 4.1: Data Acquisition
- [ ] Download Mozilla Common Voice Bengali
  - [ ] Extract train/test splits
  - [ ] Normalize format to match Lipi-Ghor
  
- [ ] Download Google FLEURS Bengali
  - [ ] Get official splits
  - [ ] Document domain differences
  
- [ ] Check IndicSUPERB benchmark
  - [ ] Identify Bengali tasks
  - [ ] Download evaluation protocols

### Task 4.2: Fine-tuning Experiments
**CRITICAL: This is the key comparison**

- [ ] **Baseline: Whisper-small pre-trained** (no fine-tuning)
  - [ ] Evaluate on held-out test set
  - [ ] Record WER/CER
  
- [ ] **Fine-tune on Mozilla Common Voice**
  - [ ] Set up training script (use Seq2SeqTrainer)
  - [ ] Train for equivalent data hours
  - [ ] Evaluate on same held-out test
  
- [ ] **Fine-tune on FLEURS**
  - [ ] Set up training script
  - [ ] Train for equivalent data hours
  - [ ] Evaluate on same held-out test
  
- [ ] **Fine-tune on Lipi-Ghor** (your dataset)
  - [ ] Use your existing fine-tuning pipeline
  - [ ] Train for same epochs/steps
  - [ ] Evaluate on held-out test
  
- [ ] Create comparison table: WER/CER across all conditions
- [ ] Statistical significance testing (paired t-test or bootstrap)

### Task 4.3: Cross-corpus Evaluation
- [ ] Train on Lipi-Ghor, test on Common Voice
- [ ] Train on Common Voice, test on Lipi-Ghor
- [ ] Analyze domain shift effects
- [ ] Document YouTube vs. read-speech differences

---

## Phase 5: Human Evaluation (Weeks 4-5 - April 20-May 3)

### Task 5.1: Sample Selection
- [ ] Randomly sample 100-200 utterances from test set
- [ ] Stratify by: domain, speaker, audio quality
- [ ] Prepare annotation interface (simple web form or spreadsheet)

### Task 5.2: Annotation Protocol
- [ ] Define rating scale (e.g., 1-5 for accuracy)
- [ ] Create annotation guidelines document
- [ ] Recruit 2-3 native Bengali speakers from team/BUET
- [ ] Run pilot with 10 utterances to calibrate

### Task 5.3: Annotation Execution
- [ ] Each annotator labels all samples independently
- [ ] Track time per utterance
- [ ] Store annotations in structured format

### Task 5.4: Analysis
- [ ] Compute inter-annotator agreement (Cohen's κ or Fleiss' κ)
- [ ] Compare human ratings with WER scores
- [ ] Identify systematic human-machine disagreements
- [ ] Select qualitative examples for paper

---

## Phase 6: Results Compilation (Week 6 - May 4-10)

### Task 6.1: Create Master Results Table
- [ ] ASR Benchmarking Results
  ```
  Model              | WER (%) | CER (%) | Inference Time
  -------------------|---------|---------|---------------
  Whisper-v3         |         |         |
  MMS 1B             |         |         |
  SeamlessM4T v2     |         |         |
  IndicWhisper       |         |         |
  ```

- [ ] Diarization Results
  ```
  System             | DER (%) | FA (%) | Miss (%) | Conf (%)
  -------------------|---------|--------|----------|----------
  pyannote 3.x       |         |        |          |
  NeMo MSDD          |         |        |          |
  ```

- [ ] Comparative Fine-tuning Results
  ```
  Training Data      | WER (%) | CER (%) | Data Hours
  -------------------|---------|---------|------------
  None (baseline)    |         |         | 0
  Common Voice       |         |         | ~X
  FLEURS             |         |         | ~Y
  Lipi-Ghor          |         |         | 882
  ```

### Task 6.2: Generate Figures
- [ ] WER comparison bar chart
- [ ] DER breakdown stacked bar chart
- [ ] Fine-tuning learning curves (if tracked)
- [ ] Domain distribution visualization (already have from React dashboard?)
- [ ] Error type distribution

### Task 6.3: Prepare Supplementary Materials
- [ ] Dataset statistics appendix
- [ ] Detailed error analysis tables
- [ ] Audio samples for review (if submission allows)
- [ ] Annotation guidelines document

---

## Phase 7: Paper Writing (Weeks 6-8 - May 4-25)

### Task 7.1: Update arXiv Paper Structure
**Note: Nishat will help heavily with this, but start the skeleton**

- [ ] **Introduction** (Sanjid - draft)
  - [ ] Motivation: lack of large-scale Bengali ASR data
  - [ ] Lipi-Ghor contribution: scale, diarization, domain coverage
  - [ ] Key findings from benchmarking
  
- [ ] **Related Work** (Team - divide sections)
  - [ ] Bengali ASR datasets
  - [ ] Multilingual speech corpora
  - [ ] Diarization datasets
  
- [ ] **Data Collection & Annotation** (Risalat - lead)
  - [ ] YouTube selection methodology
  - [ ] Diarization pipeline (pyannote)
  - [ ] Transcription approach (JSON3 captions, ASR alignment)
  - [ ] Quality control measures
  
- [ ] **Dataset Characteristics** (Fuad - lead)
  - [ ] Speaker statistics
  - [ ] Domain distribution
  - [ ] Duration statistics
  - [ ] Comparison with existing corpora
  
- [ ] **Benchmarking Experiments** (Bayazid - lead)
  - [ ] Experimental setup
  - [ ] Baseline model descriptions
  - [ ] Evaluation metrics
  - [ ] Implementation details
  
- [ ] **Results** (Sanjid - compile)
  - [ ] ASR benchmarking results
  - [ ] Diarization results
  - [ ] Fine-tuning comparisons
  - [ ] Human evaluation
  - [ ] Error analysis
  
- [ ] **Discussion** (Sanjid + Nishat)
  - [ ] Lipi-Ghor vs existing datasets
  - [ ] Limitations
  - [ ] Future work
  
- [ ] **Conclusion** (Sanjid - draft)

### Task 7.2: Writing Workflow
- [ ] Set up Overleaf project (or LaTeX repo)
- [ ] Share access with Nishat
- [ ] Weekly drafts: May 4, 11, 18
- [ ] Final revision window: May 19-24

---

## Phase 8: Submission Prep (Week 8 - May 19-25)

### Task 8.1: ARR Submission Requirements
- [ ] Check ARR author guidelines
- [ ] Prepare anonymized submission (remove author info, affiliations)
- [ ] Format references (ACL style)
- [ ] Check page limits (typically 8 pages + references)

### Task 8.2: Final Checks
- [ ] Proofread entire paper
- [ ] Verify all tables/figures cited
- [ ] Check reproducibility: code/data availability statements
- [ ] Prepare ethics statement if required
- [ ] Submit to ARR by **May 25, 2026**

---

## Coordination & Communication

### Weekly Sync Meetings
- [ ] **Every Monday 9 PM (BD time)** - Team Villagers standup
  - Progress updates
  - Blocker identification
  - Task redistribution if needed

### Checkpoints with Nishat
- [ ] **Early May (Week 6)** - Share preliminary results
  - Email summary of WER/CER/DER findings
  - Highlight any surprising results
  - Get feedback on framing

### Documentation
- [ ] **Shared Google Doc** - Running notes on experiments
- [ ] **GitHub Repo** - All evaluation scripts, results CSVs
- [ ] **HuggingFace Model Hub** - Upload fine-tuned models
- [ ] **Results Spreadsheet** - Master tracking of all metrics

---

## Risk Mitigation

### High-Risk Items
1. **Fine-tuning experiments take too long**
   - Mitigation: Use Whisper-small (faster), reduce epochs if needed
   - Fallback: Focus on pre-trained benchmarking only
   
2. **NeMo MSDD setup issues**
   - Mitigation: Start early, allocate buffer time
   - Fallback: Use only pyannote if NeMo fails
   
3. **Human evaluation recruitment**
   - Mitigation: Start recruiting annotators now
   - Fallback: Use team members + 1-2 external annotators

4. **Computing resources**
   - Mitigation: Use Kaggle GPUs, Colab Pro if needed
   - Budget: Allocate $50-100 for compute if necessary

---

## Immediate Next Steps (This Weekend/Monday)

**Priority 1 (Start Saturday/Sunday):**
1. Create GitHub repo for evaluation pipeline
2. Download Whisper-v3, MMS, SeamlessM4T, IndicWhisper
3. Prepare test set split from Lipi-Ghor
4. Set up WER/CER computation scripts

**Priority 2 (Monday morning):**
1. Team meeting - assign model ownership
2. Start first inference runs (Whisper-v3)
3. Set up results tracking spreadsheet
4. Email Nishat confirming receipt + initial timeline

**Priority 3 (This week):**
1. Complete all ASR model setups
2. Run initial benchmarks
3. Start diarization tool setup

---

**This is aggressive but doable with focused execution. The key is parallel work across the team and starting the compute-heavy tasks (fine-tuning, inference) immediately.**
