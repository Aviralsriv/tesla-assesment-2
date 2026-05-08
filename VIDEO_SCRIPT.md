# 🎬 Video Script — Tesla Multimodal SSL Project
**Author:** Aviral Srivastava
**Estimated Duration:** 5–7 minutes
**Tone:** Confident, clear, technical but approachable

---

## 🎙️ SCENE 1 — INTRO (0:00 – 0:30)
> *[Camera on face or screen recording of GitHub repo]*

"Hi, I'm Aviral Srivastava, and in this video I'll be walking you through
my Tesla Assessment project — a Multimodal Self-Supervised Learning model
for autonomous vehicle log understanding.

The core idea is this: Tesla vehicles generate massive amounts of sensor
data every single day — camera footage, LiDAR scans, CAN bus signals, GPS
traces, and HD map graphs. And the challenge is — most of this data has
NO labels. So how do you train a powerful AI model on it?

The answer is: Self-Supervised Learning. And that's exactly what I've built."

---

## 🎙️ SCENE 2 — PROBLEM STATEMENT (0:30 – 1:00)
> *[Show the README architecture diagram or documentation PDF]*

"Traditional supervised learning requires expensive human annotation.
For autonomous driving, labelling even a single hour of data can take hundreds
of man-hours.

My approach bypasses this entirely. The model learns purely from the
structure of the data itself — no labels required during pretraining.

Once pretrained, the same model can be fine-tuned for critical downstream
tasks like accident detection, driver behaviour classification, and trajectory
prediction — achieving strong results with minimal labelled data."

---

## 🎙️ SCENE 3 — ARCHITECTURE WALKTHROUGH (1:00 – 2:30)
> *[Show architecture diagram from documentation or README]*

"Let me walk you through the architecture.

**Step 1 — Data Pipeline.**
The pipeline.py file handles loading of HDF5 vehicle logs. Each segment
contains five modalities: camera images, LiDAR point clouds, CAN bus
signals, GPS coordinates, and HD-Map graph data.

We apply a DualViewAugment — creating two differently augmented versions
of the same segment. This is the foundation of contrastive learning.

**Step 2 — Modality Encoders.**
Each sensor modality has its own dedicated encoder:
- Camera frames go through DINOv2 ViT-B, a state-of-the-art vision transformer, producing a 768-dim vector.
- LiDAR point clouds are processed by a PointNet MLP into 1024 dimensions.
- CAN bus signals go through a Bidirectional LSTM — great for time-series data.
- GPS coordinates use a lightweight Transformer encoder.
- The HD-Map lane graph is processed by a 3-layer Graph Convolutional Network.

**Step 3 — Fusion.**
All five encodings are projected to a unified 1024-dim space and fed into a
FusionTransformer — which uses cross-attention to capture relationships between
modalities. The output is a single 2048-dimensional fused embedding representing
the full scene."

---

## 🎙️ SCENE 4 — TRAINING STRATEGY (2:30 – 3:30)
> *[Show train.py briefly or the loss equations from documentation]*

"Now let's talk about how the model is trained.

I use a **DINOv2-style self-supervised objective** with two components:

First — NT-Xent loss, also known as InfoNCE. For a pair of views from the same
segment, we maximize their embedding similarity while pushing apart embeddings from
different segments. This forces the model to learn what makes each driving scene
unique.

Second — a DINO loss against an EMA teacher. The teacher is a slow-moving copy
of the student network — its weights update via exponential moving average, not
backpropagation. This provides stable training targets and prevents collapse.

The total loss combines both objectives across individual modalities AND the
fused embedding.

For the optimizer, I use AdamW with cosine learning rate decay and a 5% linear
warmup. Gradient clipping at 3.0 ensures training stability."

---

## 🎙️ SCENE 5 — CODE DEMO (3:30 – 4:30)
> *[Screen recording — show project folder, then open GitHub repo]*

"Here's the project structure on GitHub.

- pipeline.py handles the entire data loading and augmentation.
- models/encoders.py contains all five modality encoders.
- models/fusion.py has the FusionTransformer and EMA update logic.
- models/losses.py implements NT-Xent and the DINO loss.
- train.py ties everything together — checkpointing, LR scheduling, logging.

[Open train.py]

You can see the training command is clean and simple — pass the HDF5 data
path, set epochs and batch size, and training begins. The model automatically
resumes from checkpoints if interrupted.

[Show checkpoints folder]

After 20 epochs of training, we have per-epoch checkpoints saved here —
each around 1.3 GB, storing student weights, teacher weights, and full
optimizer state."

---

## 🎙️ SCENE 6 — EVALUATION (4:30 – 5:00)
> *[Show eval.py or the results table from documentation]*

"After pretraining, the model can be evaluated in two ways.

A **linear probe** — where we freeze the pretrained encoder and train only
a single linear layer on top — for tasks like accident detection. The target
is AUROC above 0.90.

And a **trajectory prediction head** — where the model predicts the vehicle's
future 30-step path. We measure Average Displacement Error and Final Displacement
Error, targeting under 1.5 and 3.0 metres respectively.

These downstream metrics validate that the SSL representations are genuinely
useful — not just low-loss artefacts."

---

## 🎙️ SCENE 7 — COLAB & REPRODUCIBILITY (5:00 – 5:30)
> *[Show Colab notebook or mention it]*

"For reproducibility, I've included a Google Colab notebook that:
- Installs all dependencies automatically
- Clones the GitHub repo
- Generates a synthetic dataset in the exact same HDF5 format
- And trains the model end-to-end in under 15 minutes on a free T4 GPU

No manual data upload required — everything is automated."

---

## 🎙️ SCENE 8 — CLOSING (5:30 – 6:00)
> *[Camera back on face or final GitHub view]*

"To summarise:

I've built a complete multimodal self-supervised learning pipeline for
autonomous vehicle data — from raw sensor ingestion to a fused 2048-dim
scene representation — trained with state-of-the-art contrastive losses
and evaluated on safety-critical downstream tasks.

The full code, documentation, and Colab notebook are available on GitHub.

Thank you for watching — I'm happy to answer any questions."

---

## 📋 RECORDING TIPS

| Tip | Detail |
|---|---|
| **Screen sections** | Show GitHub repo, README diagram, train.py, checkpoints folder |
| **Keep it moving** | Don't dwell more than 30s on any single screen |
| **Voice pace** | Speak slightly slower than normal — evaluators need to follow |
| **Confidence** | You built this — own it. Use "I designed", "I implemented" |
| **Background** | Clean desktop, close unnecessary windows |
| **Tools** | OBS Studio (free) or Loom for screen recording |

---

*Total estimated time: 5–6 minutes | Script word count: ~750 words*
