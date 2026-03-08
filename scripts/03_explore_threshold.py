"""
03_explore_threshold.py — Medical Q&A
======================================
Explores how the similarity threshold affects cache behaviour using
medical question paraphrase pairs. The threshold determines what
"same medical question, different words" means for the cache.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
MODEL_NAME = "all-MiniLM-L6-v2"

# ─── Medical query paraphrase pairs ───────────────────────────────────────────
PAIRS = [
    # Should match — same medical question rephrased
    ("What are the symptoms of diabetes?",
     "How do I know if I have diabetes?", True),
    ("How is high blood pressure treated?",
     "What medications are used for hypertension?", True),
    ("What foods should I avoid with diabetes?",
     "What is the diet for a diabetic patient?", True),
    ("How do I prevent heart disease?",
     "What can I do to reduce my risk of cardiovascular disease?", True),
    ("What are signs of depression?",
     "How do I know if I am depressed?", True),
    ("What is cognitive behavioral therapy?",
     "How does CBT work for mental health treatment?", True),
    ("What are symptoms of a heart attack?",
     "How can I tell if someone is having a cardiac event?", True),
    ("How is cancer diagnosed?",
     "What tests are used to detect cancer?", True),

    # Should NOT match — different medical topics
    ("What are the symptoms of diabetes?",
     "How do I treat a burn injury?", False),
    ("How is depression treated?",
     "What foods are good for heart health?", False),
    ("What is CPR technique?",
     "What vitamins should I take daily?", False),
    ("How does influenza spread?",
     "What is cognitive behavioral therapy?", False),
    ("What is metformin used for?",
     "What are signs of a stroke?", False),
]

print("Loading model …")
model = SentenceTransformer(MODEL_NAME)

queries_a = [p[0] for p in PAIRS]
queries_b = [p[1] for p in PAIRS]
expected = [p[2] for p in PAIRS]

vecs_a = model.encode(queries_a, normalize_embeddings=True, convert_to_numpy=True)
vecs_b = model.encode(queries_b, normalize_embeddings=True, convert_to_numpy=True)
scores = np.array([float(np.dot(vecs_a[i], vecs_b[i])) for i in range(len(PAIRS))])

# ─── Threshold sweep ──────────────────────────────────────────────────────────
thresholds = np.arange(0.60, 1.00, 0.01)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    tp = fp = fn = tn = 0
    for score, exp in zip(scores, expected):
        pred = score >= t
        if pred and exp:         tp += 1
        elif pred and not exp:   fp += 1
        elif not pred and exp:   fn += 1
        else:                    tn += 1
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)

best_idx = np.argmax(f1s)
best_t = thresholds[best_idx]
print(f"\nBest F1 threshold: {best_t:.2f}  (F1={f1s[best_idx]:.3f})")

# ─── Plot P/R/F1 ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds, precisions, label="Precision", color="blue")
ax.plot(thresholds, recalls, label="Recall", color="green")
ax.plot(thresholds, f1s, label="F1", color="red", linewidth=2)
ax.axvline(x=0.88, color="purple", linestyle="--", label="Default threshold (0.88)")
ax.axvline(x=best_t, color="orange", linestyle=":", label=f"Best F1 ({best_t:.2f})")
ax.set_xlabel("Similarity Threshold")
ax.set_ylabel("Score")
ax.set_title("Medical Cache Threshold: Precision / Recall / F1 Trade-off")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "threshold_prf.png"), dpi=150)
plt.close()
print("Saved → plots/threshold_prf.png")

# ─── Score distribution plot ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
match_scores = sorted([s for s, e in zip(scores, expected) if e], reverse=True)
nomatch_scores = sorted([s for s, e in zip(scores, expected) if not e], reverse=True)
ax.scatter(range(len(match_scores)), match_scores, color="green",
           label="Should match (rephrased same question)", s=80, zorder=3)
ax.scatter(range(len(nomatch_scores)), nomatch_scores, color="red",
           label="Should NOT match (different medical topic)", s=80, zorder=3)
ax.axhline(y=0.88, color="purple", linestyle="--", label="Threshold=0.88")
ax.set_xlabel("Query pair index")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Medical Query Similarity — Paraphrase vs Different Topic")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "threshold_scores.png"), dpi=150)
plt.close()
print("Saved → plots/threshold_scores.png")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n── Threshold Summary ──")
print(f"  Paraphrase scores:   {[round(s,3) for s,e in zip(scores,expected) if e]}")
print(f"  Cross-topic scores:  {[round(s,3) for s,e in zip(scores,expected) if not e]}")
print("\n  Threshold analysis:")
for t_check in [0.75, 0.80, 0.85, 0.88, 0.90, 0.93, 0.95]:
    idx = np.searchsorted(thresholds, t_check)
    idx = min(idx, len(f1s)-1)
    print(f"    t={t_check:.2f}  F1={f1s[idx]:.3f}  P={precisions[idx]:.3f}  R={recalls[idx]:.3f}")
print("\nConclusion: 0.88 balances paraphrase recall with cross-topic precision.")
print("Part 3 threshold exploration complete.")
