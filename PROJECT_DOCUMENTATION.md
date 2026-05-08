# Federated Deepfake Detection

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [How Do We Solve It?](#2-how-do-we-solve-it)
3. [System Architecture](#3-system-architecture)
4. [Feature Extraction — What the Models Actually See](#4-feature-extraction--what-the-models-actually-see)
5. [The Three Training Stages](#5-the-three-training-stages)
6. [Loss Functions — How the Models Learn](#6-loss-functions--how-the-models-learn)
7. [Federated Learning Layer](#7-federated-learning-layer)
8. [Design Decisions and Why](#8-design-decisions-and-why)
9. [Project Structure](#9-project-structure)
10. [Configuration Reference](#10-configuration-reference)
11. [Running the Project](#11-running-the-project)

---

## 1. What Problem Are We Solving?

### The Deepfake Problem

Deepfake generators (StyleGAN, Stable Diffusion, FaceSwap, etc.) can create
highly realistic fake images. Detecting these fakes is critical for combating
misinformation, identity fraud, and media manipulation.

### Why Existing Detectors Fail

Most deepfake detectors are trained on images from specific generators.
They learn patterns like "StyleGAN images have these specific pixel patterns"
rather than "fake images have these universal manipulation artifacts." When
a new generator appears that they've never seen, they fail completely.

**This is the generalization problem.**

### What This Project Does

This project builds a deepfake detector that:

1. **Generalizes across generators** — detects fakes from generators it has
   never seen during training
2. **Is lightweight** — small enough to deploy on edge devices or in
   bandwidth-constrained federated settings
3. **Preserves privacy** — supports federated learning so institutions can
   collaboratively train without sharing their sensitive face datasets

---

## 2. How Do We Solve It?

We use three key ideas, each addressing a specific challenge:

### Idea 1: Teacher-Student Knowledge Distillation

> **Challenge:** We want a small, deployable model but small models are usually weak.

A large, powerful **Teacher** model learns to detect deepfakes using both
RGB images (what you see) and forensic signals (hidden manipulation traces).
Then a smaller **Student** model is trained to mimic the Teacher's knowledge —
achieving similar accuracy in a model that's ~10x smaller.

The Student intentionally does NOT see RGB images. It only sees forensic
and gradient features. This forces it to learn manipulation artifacts rather
than memorizing what faces look like.

### Idea 2: Multi-Signal Forensic Features

> **Challenge:** Pixel-level manipulation traces are subtle and easy to miss.

Instead of looking at just the RGB image, we extract 4 types of forensic
signals that deepfake generators struggle to fake perfectly:

| Signal | What It Detects | Why Generators Can't Fake It |
|--------|----------------|------------------------------|
| **SRM** (Steganalysis Rich Model) | Noise residuals from pixel manipulation | Generators introduce statistical noise patterns different from real cameras |
| **FFT** (Fast Fourier Transform) | Frequency-domain artifacts | Generators often produce images with unusual frequency distributions |
| **Wavelets** (Haar decomposition) | Multi-scale texture inconsistencies | Manipulation creates artifacts at specific spatial scales |
| **Laplacian** (Edge detection) | Blending boundary artifacts | Face-swap boundaries and inpainting edges leave detectable edge patterns |

These 4 signals are stacked into a 12-channel tensor (4 signals × 3 color channels)
and fed to the models as "forensic features."

### Idea 3: Gradient Reversal Layer (GRL) for Generator Invariance

> **Challenge:** The model might still learn generator-specific shortcuts.

Even with forensic features, the model could learn "StyleGAN fakes have this
noise pattern" instead of "all fakes have abnormal noise." The GRL prevents
this through **adversarial training**:

- A **generator classifier head** tries to identify which generator made each fake
- The GRL **reverses the gradients** flowing back to the student, forcing it
  to learn features that make generator identification *impossible*

The result: the student learns **universal forgery features** that work across
all generators, including ones it has never seen.

---

## 3. System Architecture

### Overall Pipeline

```
                    ┌─────────────────────────────────────────────────┐
                    │                 TRAINING PIPELINE               │
                    │                                                 │
                    │  Stage 1        Stage 2           Stage 3       │
                    │  ───────        ───────           ───────       │
                    │  Train          Train Student     Train Student │
                    │  Teacher        via KD from       with GRL for  │
                    │  (large)        Teacher           generator     │
                    │                 (small)           invariance    │
                    └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   Deployed Student    │
                              │   (lightweight,       │
                              │    generalizable)     │
                              └───────────────────────┘
```

### Model Architecture

```
TEACHER MODEL (11.7M params)
═══════════════════════════════════════════════
  Input Image (3×256×256)
      │
      ├──► ResNet-18 Backbone ──► 512-dim
      │    (RGB semantics)
      │
      ├──► ForensicCNN ─────────► 128-dim
      │    (12-ch forensic stack)
      │
      └──► Concatenate ──► MLP(640→512) ──► Classifier(512→2)
                                │
                           embedding (512-dim)


STUDENT MODEL (1.0M params)
═══════════════════════════════════════════════
  Input Image (3×256×256)
      │
      ├──► Forensic Stack ──► SmallCNN ──► 256-dim
      │    (12 channels)      (64→128→256)
      │
      ├──► Gradient Extractor ──► SmallCNN ──► 256-dim
      │    (3 channels)           (64→128→256)
      │
      └──► Concatenate ──► MLP(512→256) ──► Classifier(256→2)
                                │
                           embedding (256-dim)
```

### Why the Student is Smaller

| Property | Teacher | Student |
|----------|---------|---------|
| RGB backbone | ResNet-18 (11M params) | None |
| Forensic CNN | 3-layer CNN (128-dim) | 3-layer CNN (256-dim) |
| Gradient features | None | 3-layer CNN (256-dim) |
| Total parameters | 11.7M | 1.0M |
| Embedding dimension | 512 | 256 |

The student trades RGB understanding for gradient sensitivity analysis.
The Teacher uses a powerful ResNet to understand image semantics (faces,
backgrounds, textures). The Student replaces this with a gradient map that
captures "what a pre-trained ImageNet model is sensitive to" in the input —
a different but complementary signal.

---

## 4. Feature Extraction — What the Models Actually See

### Forensic Stack (`features/forensic.py`)

From a single RGB image, we compute 4 forensic feature maps:

```
Input Image (3×256×256)
    │
    ├──► SRM Filters ──────► 3×256×256  (noise residuals)
    ├──► FFT Magnitude ────► 3×256×256  (frequency spectrum)
    ├──► Haar Wavelets ────► 3×256×256  (multi-scale decomposition)
    └──► Laplacian Edge ───► 3×256×256  (edge map)
    
    Stacked output: 12×256×256
```

**SRM (Steganalysis Rich Model):** Uses 3 pre-defined high-pass convolutional
filters that extract noise residuals. Real camera images have characteristic
noise patterns; generated images have different ones.

**FFT:** Converts the image to the frequency domain. Deepfake generators
often leave artifacts in specific frequency bands that are invisible to the
human eye but detectable computationally.

**Wavelets:** Decomposes the image into 4 sub-bands (LL, LH, HL, HH) at
half resolution, then upscales back. This captures texture inconsistencies
at different spatial scales.

**Laplacian:** A second-derivative edge detector that highlights boundaries
and transitions. Face-swap operations often leave subtle blending edges.

**Why 12 channels?** Each of the 4 methods processes all 3 color channels
independently, producing 4 × 3 = 12 feature channels. This preserves
color-specific artifacts (some manipulation traces are more visible in
specific color channels).

### Gradient Features (`features/gradient.py`)

```
Input Image (3×256×256)
    │
    ▼
Frozen MobileNetV3 (pretrained on ImageNet)
    │
    ├──► Forward pass ──► logits
    │
    └──► torch.autograd.grad(logits.sum(), input) ──► gradient map (3×256×256)
```

The gradient map shows "what pixels does a pre-trained classifier find most
important?" This is useful for deepfake detection because:

- Real images activate the classifier in natural, distributed patterns
- Fake images often have concentrated or unusual activation patterns
  (especially around manipulated regions)

**Why MobileNetV3 instead of ResNet-18?** MobileNetV3-Small has 2.5M params
vs ResNet-18's 11.7M. Since the gradient extractor runs a full
forward + backward pass every batch, using a lighter model reduces the
per-batch computation by ~3-4x while producing comparable gradient signals
(both are pretrained on ImageNet).

**Why `.detach()`?** The gradient map is treated as a static feature
(like the forensic stack), not a differentiable transform. The student
learns to interpret gradient patterns but cannot backpropagate through the
extraction process. This is intentional — it prevents gradient pollution
and keeps the computational graph manageable.

---

## 5. The Three Training Stages

### Stage 1: Train the Teacher (5 epochs)

```
Goal: Build a powerful deepfake detector as a knowledge source

Input:  RGB image + 12-channel forensic stack
Model:  TeacherModel (ResNet-18 + ForensicCNN)
Losses: CrossEntropy + SupConLoss
Output: A teacher with ~99% AUC
```

The teacher sees both RGB and forensic features, giving it maximum
information. It learns strong representations that combine visual semantics
with manipulation artifacts. After training, the teacher is **frozen** —
its weights never change again.

### Stage 2: Knowledge Distillation (12 epochs)

```
Goal: Transfer the teacher's knowledge to a smaller student

Input:  12-channel forensic stack + 3-channel gradient map
Model:  StudentModel (2× SmallCNN)
Losses: 1.2×CrossEntropy + 0.1×KDLoss + 0.1×SupConLoss
Output: A student that approximates the teacher's performance
```

The student learns from three signals simultaneously:

1. **CrossEntropy (CE):** Direct classification — "is this image real or fake?"
2. **KD Loss:** Match the teacher's soft predictions — the teacher's output
   probabilities contain richer information than hard labels. For example,
   the teacher might say "95% fake, 5% real" which tells the student more
   than just "fake."
3. **SupCon Loss:** Shape the embedding space so real images cluster together
   and fake images cluster together.

**Why 1.2× on CE?** The CE loss is the primary classification objective.
The multiplier ensures it slightly dominates the gradient signal so the
student prioritizes "get the answer right" over "perfectly mimic the teacher."
This is important because the student *cannot* perfectly mimic the teacher
(it doesn't see RGB), so CE prevents it from chasing an impossible target.

### Stage 3: Generator Invariance via GRL (10 epochs)

```
Goal: Make the student's features generator-agnostic

Input:  Same as Stage 2
Model:  StudentModel + GRL + GeneratorClassifier
Losses: CE + KD + SupCon + λ_grl × AdversarialLoss
Output: A student that generalizes to unseen generators
```

This is where the magic happens. A small classifier head tries to predict
which generator made each fake image. The **Gradient Reversal Layer (GRL)**
sits between the student's embedding and this classifier:

```
Student Embedding ──► GRL ──► Generator Classifier ──► "Which generator?"
                      │
                      │ During backward pass:
                      │ gradients are REVERSED (multiplied by -λ)
                      │
                      └── Forces student to learn features that make
                          generator identification IMPOSSIBLE
```

**How GRL works mechanically:**
- Forward pass: identity function (passes embeddings unchanged)
- Backward pass: multiplies gradients by `-λ` (reverses direction)
- Effect: the student is rewarded for making the generator classifier FAIL

**λ scheduling:** The GRL strength `λ` starts small (0.05) and increases
over training. This lets the student first learn basic detection features
(epochs 1-3), then progressively forces generator invariance (epochs 4-10).

**Why only on fake images?** Real images don't have a generator, so the
adversarial loss is only computed on fake samples. The generator IDs are
remapped to 0-indexed (e.g., generators 1,2,3 → indices 0,1,2).

---

## 6. Loss Functions — How the Models Learn

### CrossEntropy Loss
Standard binary classification: real (0) vs fake (1).

### KD Loss (Knowledge Distillation)
```python
KL_Divergence(student_softmax(logits/T), teacher_softmax(logits/T)) × T²
```
Temperature `T=4.0` softens the probability distributions, revealing the
teacher's uncertainty structure. The `T²` factor compensates for the
reduced gradient magnitude at high temperatures.

**Why KL divergence instead of MSE?** KL divergence compares probability
distributions directly, which is more meaningful for classification outputs
than raw distance between logit values.

### SupCon Loss (Supervised Contrastive)
Pulls embeddings of same-class samples together and pushes different-class
embeddings apart in the representation space.

Uses the **log-sum-exp trick** for numerical stability under fp16 mixed
precision. Without this, the similarity scores (divided by temperature 0.07,
which amplifies them ~14x) can overflow fp16's max value (~65504).

### Generator Adversarial Loss
Standard CrossEntropy on generator classification, but gradients are
reversed by the GRL before reaching the student. The student "sees"
this loss as: "change your features to make generator prediction worse."

### Loss Balance (Why These Specific Weights?)

```
Stage 2: loss = 1.2×CE + 0.1×KD + 0.1×SupCon
Stage 3: loss = CE + 0.1×KD + 0.1×SupCon + λ_grl×GenLoss
```

| Weight | Value | Reasoning |
|--------|-------|-----------|
| CE multiplier | 1.2 | Makes classification the dominant objective |
| λ_kd | 0.1 | Soft guidance from teacher without overwhelming CE. Higher values (0.3) caused the student to chase the teacher's unreachable logits at the expense of classification accuracy |
| λ_supcon | 0.1 | Shapes the embedding space. Original value of 0.05 was too low — the SupCon loss barely moved during training |
| λ_grl | 0.05→1.0 | Starts small to let the student learn basic features first, then ramps up to enforce generator invariance |

---

## 7. Federated Learning Layer

### Why Federated?

Different institutions (labs, social media companies, governments) each have
their own deepfake datasets. These datasets often contain sensitive face
images that cannot be shared due to privacy regulations.

Federated learning allows them to **collaboratively train a single global
model without sharing any data**.

### Architecture

```
Phase 1: Train Teacher Centrally (one-time)
═══════════════════════════════════════════
  Public/shared dataset ──► Train Teacher ──► Freeze ──► Distribute to clients

Phase 2: Federated Student Training (N rounds)
═══════════════════════════════════════════
  
  Round 1:
  ┌─ Server broadcasts global student weights ─────────────────────┐
  │                                                                │
  │  Client 1              Client 2              Client 3          │
  │  (StyleGAN data)       (Diffusion data)      (FaceSwap data)   │
  │  KD + GRL locally      KD + GRL locally      KD + GRL locally  │
  │  3 local epochs        3 local epochs        3 local epochs    │
  │                                                                │
  │  Δw₁                   Δw₂                   Δw₃              │
  └──────────────── FedAvg Aggregation ────────────────────────────┘
                           │
                    Global Student Updated
                           │
  Round 2:
  ┌─ Server broadcasts updated weights ────────────────────────────┐
  │  ...repeat...                                                  │
  └────────────────────────────────────────────────────────────────┘
```

### FedAvg Aggregation

After each round, the server computes a weighted average of all client models:

```
w_global = Σ (nₖ / n_total) × wₖ
```

Where `nₖ` = number of samples on client k. Clients with more data have
proportionally more influence on the global model.

### FedProx Regularization

In non-IID settings (each client sees different generators), client models
can drift apart significantly. FedProx adds a penalty term:

```
L_total = L_original + (μ/2) × ‖w_local - w_global‖²
```

This pulls each client's model back toward the global model, preventing
divergence. The weight `μ=0.01` controls how strongly.

### Why GRL is Especially Powerful in Federated Settings

```
WITHOUT GRL:
  Client 1 (StyleGAN) → learns "StyleGAN-specific features"
  Client 2 (Diffusion) → learns "Diffusion-specific features"
  After FedAvg → confused model (conflicting features)

WITH GRL:
  Client 1 (StyleGAN) → learns "universal forgery features"
  Client 2 (Diffusion) → learns "universal forgery features"
  After FedAvg → coherent model (features align naturally)
```

### What Gets Federated (and What Doesn't)

| Component | Federated? | Reason |
|-----------|-----------|--------|
| Student weights | ✅ Yes | These are the core detection model |
| Teacher weights | ❌ No | Teacher is frozen and identical across all clients |
| Generator head | ❌ No | Each client sees different generators, so their heads are incompatible |
| Gradient extractor | ❌ No | Frozen pre-trained model, identical everywhere |

---

## 8. Design Decisions and Why

### Why doesn't the student see RGB?

The student intentionally uses only forensic + gradient features (no RGB).

**Reason 1 — Generalization:** RGB features include semantic content (faces,
hair, skin texture). The student might learn "this face texture is common in
fakes" which doesn't generalize. Forensic features capture manipulation
artifacts that are independent of image content.

**Reason 2 — Model size:** Adding an RGB backbone (ResNet-18) would add
~11M parameters, making the student as large as the teacher. The whole point
of KD is a smaller model.

**Reason 3 — Federation efficiency:** Smaller models mean less data
transmitted per round in federated learning.

**Trade-off:** The student can't match the teacher's performance (AUC ~0.94
vs ~0.99). This is the price of generalization and deployability.

### Why CosineAnnealingLR with eta_min=1e-5?

The learning rate follows a cosine decay schedule:

```
LR: 3e-4 → ... → 1e-5 (over T_max epochs)
```

**Why cosine?** Smoother than step decay, avoids sudden LR drops that can
destabilize training.

**Why eta_min=1e-5 instead of 0?** Without `eta_min`, the LR reaches
exactly 0 on the final epoch, wasting it entirely. A small minimum ensures
every epoch does meaningful work.

### Why batch_size=16?

The system has 3 models on GPU simultaneously during Stage 2:
- Teacher (11.7M params) — for KD targets
- Student (1.0M params) — being trained
- Gradient extractor (MobileNetV3, 2.5M params) — computing gradient maps

At batch_size=32 with the larger SmallCNN (64→128→256 channels), the
combined activation memory exceeded GPU VRAM. Batch_size=16 halves the
activation memory while maintaining training quality.

### Why gradient clipping (max_norm=1.0)?

Mixed precision training (fp16) can produce large gradient values that
cause instability. Gradient clipping caps the total gradient norm at 1.0,
preventing sudden parameter jumps. Applied to all training stages.

### Why WeightedRandomSampler?

The dataset has roughly equal real/fake samples (11012 real, 11053 fake),
but the sampler ensures perfectly balanced batches. Without it, random
sampling could produce batches that are 70% real / 30% fake (or vice versa),
which biases the model.

### Why GaussianBlur with p=0.3?

Data augmentation includes random Gaussian blur, but at only 30% probability.

**Why not always?** The forensic features (SRM, Laplacian) detect fine-grained
pixel-level artifacts. Heavy blurring destroys these artifacts, making the
forensic features useless and essentially training the model on noise.

**Why not never?** Some real-world images are blurry (low-quality cameras,
compression). The model needs some exposure to blur to be robust.

### Why separate optimizer for gen_head in Stage 3?

The generator classifier (gen_head) has its own Adam optimizer, separate from
the student's optimizer. 

**Reason:** The student optimizer carries momentum from Stage 2. If we added
gen_head parameters to the same optimizer, it would corrupt the momentum
state for the student's existing parameters. A separate optimizer keeps the
student's training momentum intact while the gen_head trains from scratch.

---

## 9. Project Structure

```
federated-deepfake/
│
├── configs/
│   └── config.yaml              # All hyperparameters
│
├── data/
│   ├── real/                    # Real images directory
│   ├── fake/                    # Fake images (subdirs per generator)
│   │   ├── stylegan/
│   │   ├── diffusion/
│   │   └── faceswap/
│   ├── dataset.py               # DeepfakeDataset class + augmentations
│   ├── loader.py                # Sample loading with file filtering
│   └── partitioner.py           # Federated data partitioning (IID/non-IID)
│
├── models/
│   ├── teacher.py               # TeacherModel (ResNet-18 + ForensicCNN)
│   ├── student.py               # StudentModel (2× SmallCNN)
│   ├── grl.py                   # Gradient Reversal Layer (autograd)
│   └── gen_classifier.py        # Generator classifier head (MLP)
│
├── features/
│   ├── forensic.py              # SRM, FFT, Wavelet, Laplacian extraction
│   └── gradient.py              # Input gradient extraction via frozen model
│
├── losses/
│   ├── losses.py                # CE, KD, SupCon, Generator Adversarial
│   └── fedprox.py               # FedProx proximal regularization
│
├── training/
│   ├── train.py                 # All 3 training stages
│   └── validate.py              # Evaluation with metrics
│
├── federation/
│   ├── server.py                # Federated server (round orchestration)
│   ├── client.py                # Federated client (local training)
│   └── aggregation.py           # FedAvg weighted averaging
│
├── utils/
│   ├── checkpoint.py            # Model save/load
│   ├── logger.py                # Loss logging
│   └── visualize.py             # t-SNE visualization
│
├── main.py                      # Centralized training entry point
└── main_federated.py            # Federated training entry point
```

---

## 10. Configuration Reference

```yaml
# config.yaml

batch_size: 16              # Images per batch (limited by GPU VRAM)
learning_rate: 0.0003       # Adam optimizer initial LR

# Loss weights
lambda_kd: 0.1              # Knowledge distillation weight
lambda_supcon: 0.1          # Supervised contrastive weight
lambda_grl: 0.05            # GRL starting strength (ramps to 1.0)

# Temperatures
temperature_kd: 4.0         # KD softmax temperature (higher = softer)
temperature_supcon: 0.07    # SupCon temperature (lower = sharper contrast)

# Federated Learning
num_clients: 3              # Number of federated clients
num_rounds: 10              # Number of federated rounds
local_epochs: 3             # Local training epochs per client per round
clients_per_round: null     # Clients sampled per round (null = all)
fedprox_mu: 0.01            # Proximal term weight (0 = pure FedAvg)
iid_partition: false        # false = non-IID (split by generator)
```

---

## 11. Running the Project

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support
- torchvision, scikit-learn, PyYAML, Pillow

### Data Setup

Place images in:
```
data/real/          ← real images (.jpg, .png, .bmp, .tiff, .webp)
data/fake/
    ├── generator1/ ← fake images from generator 1
    ├── generator2/ ← fake images from generator 2
    └── generator3/ ← fake images from generator 3
```

### Centralized Training

```bash
python main.py
```

Runs: Teacher (5 epochs) → Student KD (12 epochs) → GRL (10 epochs)

Outputs:
- `checkpoints/teacher_stage1.pth`
- `checkpoints/student_stage2.pth`
- `checkpoints/student_final.pth`
- `checkpoints/student_best.pth` (if GRL improved over KD)

### Federated Training

```bash
python main_federated.py
```

Runs: Teacher (centralized, 5 epochs) → Federated Student (10 rounds × 3 local epochs)

Outputs:
- `checkpoints/teacher_federated.pth`
- `checkpoints/student_federated_best.pth`

### Expected Results

| Model | AUC | Accuracy | Recall | Parameters |
|-------|-----|----------|--------|------------|
| Teacher | ~0.998 | ~0.98 | ~0.97 | 11.7M |
| Student (after KD) | ~0.94 | ~0.89 | ~0.78 | 1.0M |
| Student (after GRL) | ~0.95+ | ~0.90+ | ~0.80+ | 1.0M |

The ~5% AUC gap between teacher and student is the expected price of:
1. No RGB access (generalization by design)
2. 10x fewer parameters (deployability)
3. Generator-agnostic features (robustness to unseen generators)
