"""
NumNum_Whoisthere: handwritten digit recognition (MNIST) with supervised AND unsupervised voters.

Run:
    pip install gradio scikit-learn pandas pillow numpy scipy joblib matplotlib
    python digit_app_mnist_v4.py
Then open http://127.0.0.1:7860

First run downloads MNIST (~15 MB) and trains the models (a few minutes).
Everything is cached in mnist_models_v3.joblib (same file as v3, so if you already
ran v3 nothing is retrained). Delete that file if you change the models below.

Optional environment variables:
    MNIST_LOCAL_DIR  folder with the 4 MNIST .gz files (offline use)
    N_TRAIN          number of training images (default 20000)
"""
import base64
import gzip
import inspect
import io
import json
import os
import struct
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from scipy import ndimage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

CACHE_FILE = Path("mnist_models_v3.joblib")
N_TRAIN = int(os.environ.get("N_TRAIN", 20000))

UNSUPERVISED = ("K-Means", "Gaussian Mixture", "Agglomerative (Ward)")


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist():
    """Returns X_train, y_train, X_test, y_test with pixels scaled to 0-1."""
    local = os.environ.get("MNIST_LOCAL_DIR")
    if local:
        d = Path(local)
        Xtr = _read_idx(d / "train-images-idx3-ubyte.gz").reshape(-1, 784)
        ytr = _read_idx(d / "train-labels-idx1-ubyte.gz")
        Xte = _read_idx(d / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
        yte = _read_idx(d / "t10k-labels-idx1-ubyte.gz")
    else:
        print("Downloading MNIST (first run only)...")
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
        y = y.astype(int)
        Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
    return Xtr / 255.0, ytr.astype(int), Xte / 255.0, yte.astype(int)


class ClusterClassifier:
    """An unsupervised clustering model used as a digit classifier.

    Step 1: clusters are discovered from pixels alone. Labels are never used here.
    Step 2: each cluster is named after the digit that is most common inside it.
    A drawing is then assigned to clusters (softly) and inherits their names.
    """

    def __init__(self, kind, n_clusters, n_components=40):
        self.kind, self.k, self.n_components = kind, n_clusters, n_components

    def fit(self, X, y):
        self.pca = PCA(self.n_components, random_state=42).fit(X)
        Z = self.pca.transform(X)
        self.temp = None

        if self.kind == "kmeans":
            self.centers_ = KMeans(self.k, n_init=3, random_state=42).fit(Z).cluster_centers_
        elif self.kind == "gmm":
            self.gmm = GaussianMixture(
                self.k, covariance_type="diag", reg_covar=1e-3, random_state=42
            ).fit(Z)
            self.centers_ = self.gmm.means_
        else:
            sub = np.random.RandomState(42).permutation(len(Z))[:4000]
            lab = AgglomerativeClustering(self.k, linkage="ward").fit_predict(Z[sub])
            self.centers_ = np.array([Z[sub][lab == c].mean(axis=0) for c in range(self.k)])

        if self.kind != "gmm":
            self.temp = 0.1 * self._sqdist(Z).min(axis=1).mean()

        counts = np.full((self.k, 10), 0.5)
        np.add.at(counts, (self._responsibilities(Z).argmax(axis=1), y), 1)
        self.cluster_digit = counts.argmax(axis=1)
        self.cluster_probs = counts / counts.sum(axis=1, keepdims=True)
        return self

    def _sqdist(self, Z):
        C = self.centers_
        return np.maximum((Z**2).sum(1, keepdims=True) - 2 * Z @ C.T + (C**2).sum(1), 0)

    def _responsibilities(self, Z):
        if self.kind == "gmm":
            return self.gmm.predict_proba(Z)
        d2 = self._sqdist(Z)
        w = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / self.temp)
        return w / w.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._responsibilities(self.pca.transform(X)) @ self.cluster_probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def prototype(self, X):
        """The closest cluster's average image, its id, and the digit it was named after."""
        r = self._responsibilities(self.pca.transform(X))[0]
        c = int(r.argmax())
        img = self.pca.inverse_transform(self.centers_[c : c + 1])[0].reshape(28, 28)
        img = img.clip(0, None)
        return img / max(img.max(), 1e-6), c, int(self.cluster_digit[c])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "KNN": make_pipeline(PCA(50, random_state=42), KNeighborsClassifier(3, weights="distance")),
        "SVM (RBF)": make_pipeline(
            PCA(50, random_state=42),
            CalibratedClassifierCV(SVC(C=5, gamma="scale"), ensemble=False, cv=3),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
        "Neural Net (MLP)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, random_state=42),
        "K-Means": ClusterClassifier("kmeans", 64),
        "Gaussian Mixture": ClusterClassifier("gmm", 32),
        "Agglomerative (Ward)": ClusterClassifier("ward", 40),
    }


def accuracy_table(models, X_test, y_test):
    """Test accuracy of each model and of the voting ensembles."""
    rows, probs_by = [], {}
    for name, model in models.items():
        probs = model.predict_proba(X_test)
        probs_by[name] = probs
        kind = "Unsupervised" if name in UNSUPERVISED else "Supervised"
        rows.append((name, kind, (probs.argmax(axis=1) == y_test).mean()))

    preds = np.array([p.argmax(axis=1) for p in probs_by.values()])
    hard = np.array([Counter(col).most_common(1)[0][0] for col in preds.T])

    def soft(names):
        return np.mean([probs_by[n] for n in names], axis=0).argmax(axis=1)

    sup = [n for n in models if n not in UNSUPERVISED]
    ensembles = [
        ("Hard voting (all)", hard),
        ("Soft voting (all)", soft(list(models))),
        ("Soft voting (supervised only)", soft(sup)),
        ("Soft voting (unsupervised only)", soft(list(UNSUPERVISED))),
    ]
    rows += [(n, "Ensemble", (p == y_test).mean()) for n, p in ensembles]
    return pd.DataFrame(rows, columns=["Model", "Type", "Accuracy"])


def get_artifacts():
    if CACHE_FILE.exists():
        return joblib.load(CACHE_FILE)

    X_train, y_train, X_test, y_test = load_mnist()
    idx = np.random.RandomState(42).permutation(len(X_train))[:N_TRAIN]
    Xs, ys = X_train[idx], y_train[idx]

    models = build_models()
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(Xs, ys)
    print(f"Evaluating on {len(X_test):,} MNIST test images...")
    acc = accuracy_table(models, X_test, y_test)

    pca2 = PCA(2, random_state=42).fit(Xs)
    art = {
        "models": models,
        "accuracy": acc,
        "pca2": pca2,
        "map_xy": pca2.transform(Xs[:2500]),
        "map_y": ys[:2500],
    }
    joblib.dump(art, CACHE_FILE)
    return art


ART = get_artifacts()
MODELS, ACCURACY = ART["models"], ART["accuracy"]
PCA2, MAP_XY, MAP_Y = ART["pca2"], ART["map_xy"], ART["map_y"]
print(ACCURACY.to_string(index=False))


def vote(x_flat, names):
    rows, all_probs = [], []
    for name in names:
        probs = MODELS[name].predict_proba(x_flat)[0]
        pred = int(probs.argmax())
        rows.append({"name": name, "pred": pred, "conf": float(probs[pred])})
        all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)
    soft = int(np.argmax(mean_probs))
    counts = Counter(r["pred"] for r in rows)
    top = max(counts.values())
    tied = [d for d, c in counts.items() if c == top]
    hard = max(tied, key=lambda d: mean_probs[d])
    return rows, hard, soft, mean_probs, counts


def to_gray(img):
    if isinstance(img, dict):
        img = img.get("composite")
    if img is None:
        return None
    arr = np.asarray(img).astype(float)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            alpha = arr[..., 3:4] / 255.0
            arr = arr[..., :3] * alpha + 255.0 * (1 - alpha)
        arr = arr[..., :3].mean(axis=2)
    return arr


def preprocess(img):
    gray = to_gray(img)
    if gray is None:
        return None
    ink = 255 - gray if np.median(gray) > 127 else gray
    mask = ink > 40
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    crop = ink[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    small = Image.fromarray(crop.clip(0, 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(small, dtype=float)

    frame = np.zeros((28, 28))
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    frame[top : top + new_h, left : left + new_w] = small

    cy, cx = ndimage.center_of_mass(frame)
    frame = ndimage.shift(frame, (13.5 - cy, 13.5 - cx), order=1)
    frame = frame.clip(0, None)
    return frame / frame.max()


DIGIT_COLORS = [
    "#F43F8E", "#FB7A3C", "#E9A100", "#45B36B", "#14B8A6",
    "#0EA5E9", "#6366F1", "#8B5CF6", "#D946EF", "#64748B",
]

PALETTES = [
    ("bubblegum", "#FF5C9E"),
    ("tangerine", "#F97316"),
    ("sunshine", "#FACC15"),
    ("leaf", "#22C55E"),
    ("mint", "#14B8A6"),
    ("sky", "#0EA5E9"),
    ("ocean", "#3B82F6"),
    ("indigo", "#6366F1"),
    ("grape", "#7C3AED"),
    ("slate", "#64748B"),
]

ALGORITHM_INFO = {
    "Logistic Regression": {
        "steps": [
            "The 28 by 28 drawing is converted into 784 pixel numbers.",
            "During training, the model learns one weight for every pixel and digit combination.",
            "For a new drawing, it combines those weights into ten numerical scores.",
            "The softmax-style probabilities turn the scores into values that add to 100 percent.",
            "The digit with the largest probability becomes the prediction.",
        ],
        "example": "Imagine three pixels in the upper loop of a 8. During training, dark values in those pixels repeatedly appear with label 8. The model learns positive weights for that evidence and negative weights for competing digits.",
        "info": "Logistic regression is a supervised linear classifier. Despite its name, it is used for classification rather than predicting a continuous number.",
        "strength": "It is fast, stable, and gives probabilities that are easy to compare.",
        "limits": "A straight-line score can miss highly curved or unusually drawn shapes unless the useful features are already easy to separate.",
    },
    "KNN": {
        "steps": [
            "The drawing is converted to one vector containing all 784 pixels.",
            "The model compares that vector with the training drawings it has stored.",
            "A distance value measures how different two drawings are.",
            "It finds the three nearest training examples.",
            "Closer neighbors receive more voting power, and the winning digit is returned.",
        ],
        "example": "If the two nearest examples are 4 and the next is 9, weighted KNN gives 4 a larger share of the vote because the 4 examples are closer.",
        "info": "KNN, or K-nearest neighbors, learns by keeping examples instead of building a compact formula. It makes a decision only when a new drawing arrives.",
        "strength": "It is intuitive and can recognize shapes that resemble known training examples.",
        "limits": "It becomes slower and memory-heavy for large datasets, and unrelated differences such as position or stroke thickness can affect distance.",
    },
    "SVM (RBF)": {
        "steps": [
            "PCA compresses 784 pixels into the 50 directions that preserve the most useful variation.",
            "The RBF kernel compares pairs of drawings in a higher-dimensional feature space.",
            "The model learns support vectors, the training examples that lie closest to difficult boundaries.",
            "It constructs curved boundaries that separate pairs or groups of digits.",
            "Calibration converts the distance-based decision scores into useful probabilities.",
        ],
        "example": "A wide 0 and a narrow 6 may overlap in basic pixel position. A curved RBF boundary can distinguish them while still allowing slightly imperfect drawings.",
        "info": "The radial basis function SVM looks for flexible boundaries rather than one straight dividing line.",
        "strength": "It can capture complex nonlinear shapes and often performs very well on smaller, carefully prepared datasets.",
        "limits": "Training is expensive, difficult boundaries can be sensitive to parameters, and the model is harder to interpret than a tree.",
    },
    "Decision Tree": {
        "steps": [
            "The training algorithm examines possible pixel thresholds.",
            "It chooses a threshold that best separates the remaining digit groups.",
            "It repeats this process for each branch.",
            "A new drawing follows the matching yes-or-no branches.",
            "The leaf containing the most likely digit supplies the answer.",
        ],
        "example": "A first question might ask whether the rightmost pixels are dark. A later question might check whether the center is empty, helping distinguish 1, 4, and 7.",
        "info": "A decision tree represents learning as a sequence of simple questions. There is no need to multiply every pixel by a learned weight.",
        "strength": "The structure is easy to follow and can handle nonlinear patterns.",
        "limits": "One tree can overfit small visual details and become unstable when the training data changes slightly.",
    },
    "Random Forest": {
        "steps": [
            "The model creates 150 training samples by sampling the training data.",
            "Each tree is trained on a random subset of the pixels and examples.",
            "A different drawing follows a different path through every tree.",
            "Each tree reports a digit and a confidence.",
            "The forest averages the probability estimates and selects the strongest digit.",
        ],
        "example": "If 110 trees favor 3 but 40 prefer 5, the average probability for 3 can still win even if a few trees strongly prefer 5.",
        "info": "A random forest is an ensemble of decision trees. Randomness reduces the chance that every tree repeats the same mistake.",
        "strength": "It is usually more accurate and stable than a single decision tree.",
        "limits": "It uses much more memory and computation, and its final decision is harder to explain than one tree.",
    },
    "Neural Net (MLP)": {
        "steps": [
            "784 pixel values enter the network as input neurons.",
            "A hidden layer of 128 neurons combines these values into learned visual features.",
            "Later transformations add nonlinear behavior, allowing the network to represent curves and stroke relationships.",
            "Training adjusts every connection to reduce prediction errors.",
            "The output layer produces ten probabilities, one for each digit.",
        ],
        "example": "One neuron might learn to respond to a horizontal top stroke. A later neuron could combine that response with a right-side curve, making a pattern more specific to 9.",
        "info": "An MLP, or multilayer perceptron, is a small neural network made of connected processing units.",
        "strength": "It can learn combinations of features that a linear classifier cannot represent directly.",
        "limits": "It needs careful training and can be harder to explain. It may learn dataset shortcuts rather than general visual rules.",
    },
    "K-Means": {
        "steps": [
            "PCA reduces every training image to 40 features.",
            "K-Means places 64 temporary centers in that feature space.",
            "Each image is assigned to its nearest center.",
            "The centers move toward the average position of their assigned images.",
            "After training, every cluster is named with the labeled digit found most often inside it.",
        ],
        "example": "Many round drawings gather in one cluster. If 92 percent of that cluster is labeled 0, drawings assigned to the cluster become votes for 0.",
        "info": "K-Means discovers structure in pixels alone. Digit labels are used only after clusters have formed.",
        "strength": "It is fast and creates compact groups that are easy to visualize.",
        "limits": "It assumes roughly circular clusters and its groups may mix several similar-looking digits.",
    },
    "Gaussian Mixture": {
        "steps": [
            "PCA first represents every drawing using 40 compact features.",
            "A Gaussian mixture defines 32 soft probability distributions.",
            "For each image, the model calculates a probability of belonging to every distribution.",
            "Training adjusts both the position and spread of each distribution.",
            "The distribution probabilities are combined with the dominant digit learned for each cluster.",
        ],
        "example": "A drawing could receive 70 percent responsibility from a cluster dominated by 8 and 30 percent from a 9 cluster. The model therefore keeps uncertainty instead of choosing one hard cluster.",
        "info": "A Gaussian mixture uses ellipsoid-shaped probability regions with soft membership.",
        "strength": "It represents ambiguity and clusters that overlap or have different sizes.",
        "limits": "It can be sensitive to initialization and may produce unstable components for high-dimensional MNIST data.",
    },
    "Agglomerative (Ward)": {
        "steps": [
            "PCA represents a sample of 4,000 training drawings.",
            "Every drawing begins in its own cluster.",
            "Ward's method repeatedly merges the pair whose merge adds the least within-cluster variation.",
            "The process continues until 40 groups remain.",
            "Cluster centroids are used to assign a new drawing and are named by their dominant digit.",
        ],
        "example": "Several tiny groups of slightly tilted 1 drawings may merge. Together they form a larger cluster named 1.",
        "info": "Agglomerative clustering is hierarchical: it builds groups from small pieces upward.",
        "strength": "It can reveal nested structure and does not require the cluster centers to be round.",
        "limits": "The complete training computation is large, so this app uses a training subset and cluster centroids for later predictions.",
    },
    "Hard voting (all)": {
        "steps": [
            "Run every selected model, including supervised and unsupervised models.",
            "Record the single digit predicted by each model.",
            "Count how many models chose every digit.",
            "The digit with the largest number of votes wins.",
            "If several digits tie, average confidence is used to break the tie.",
        ],
        "example": "If six models say 2, four say 7, and one says 1, the hard vote is 2 regardless of how confident each model was.",
        "info": "Hard voting gives every model one vote. It is often easy to understand but ignores differences in confidence.",
        "strength": "A clear majority can reduce isolated model mistakes.",
        "limits": "A very confident wrong model has the same vote as a weak uncertain model.",
    },
    "Soft voting (all)": {
        "steps": [
            "Collect the probability distribution from every model.",
            "Align probabilities so every model represents the same ten digits.",
            "Average the probabilities for each digit.",
            "Compare the ten averages.",
            "The digit with the highest average probability wins.",
        ],
        "example": "Two models may give 3 about 60 percent and 90 percent, while three give 8 about 70 percent. The average decides the result.",
        "info": "Soft voting combines confidence rather than only final answers. It uses every model's calibration in the total.",
        "strength": "It can favor a consistently strong choice and can be more informative than hard voting.",
        "limits": "Poorly calibrated probabilities can cause one overconfident model to dominate the average.",
    },
    "Soft voting (supervised only)": {
        "steps": [
            "Select all currently chosen supervised models.",
            "Collect each model's probability for digits 0 through 9.",
            "Average those probabilities by digit.",
            "Ignore unsupervised model probabilities in this vote.",
            "Choose the digit with the largest supervised average.",
        ],
        "example": "The tree and neural network may strongly support 5 while a linear model supports 3. Their average can reveal which choice is most consistent.",
        "info": "This ensemble combines supervised learners that were trained directly with digit labels.",
        "strength": "It compares only models that learned labeled examples in similar ways.",
        "limits": "It shares the same supervised training biases and may repeat a common training mistake.",
    },
    "Soft voting (unsupervised only)": {
        "steps": [
            "Select the three unsupervised cluster models.",
            "Collect each cluster model's probability distribution.",
            "Average the probabilities for every digit.",
            "Ignore supervised models in this vote.",
            "Return the digit with the largest average.",
        ],
        "example": "Two clusters might strongly support 6 while a Gaussian mixture gives a softer vote to 0. The average shows how the unsupervised evidence combines.",
        "info": "This vote uses models that discover pixel groups before labels are assigned to clusters.",
        "strength": "It can reveal agreement based on visual structure rather than direct label learning.",
        "limits": "Cluster names can be misleading when several digits share a visually similar cluster.",
    },
}

_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>'
ICON_MOON = _SVG.format('<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>')
ICON_SUN = _SVG.format(
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'
)
ICON_INFO = _SVG.format('<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>')
ICON_X = _SVG.format('<path d="M6 6l12 12M18 6L6 18"/>')


def tile(d, size="sm"):
    return f'<span class="tile {size}" style="--c:{DIGIT_COLORS[d]}">{d}</span>'


def pill(name):
    kind = "uns" if name in UNSUPERVISED else "sup"
    text = "Unsupervised" if kind == "uns" else "Supervised"
    return f'<span class="pill {kind}">{text}</span>'


def message_card(title, text):
    return f'<div class="empty"><div class="empty-title">{title}</div><div class="empty-text">{text}</div></div>'


def png_uri(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def data_uri(arr, px, resample):
    pixels = (255 - np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return png_uri(Image.fromarray(pixels).resize((px, px), resample))


def header_html():
    n = len(MODELS)
    swatches = "".join(
        f'<button class="sw" data-palette="{name}" style="--s:{color}" title="{name.title()} palette" '
        f'aria-label="{name.title()} palette" aria-pressed="{"true" if name == "bubblegum" else "false"}">{index}</button>'
        for index, (name, color) in enumerate(PALETTES)
    )
    return (
        '<header class="top"><div class="brand"><h1>NumNum ... Who is there?</h1>'
        f"<p>Draw a digit and see how {n} algorithms read it.</p></div>"
        '<div class="controls">'
        f'<div class="swatches" role="group" aria-label="Color palette">{swatches}</div>'
        '<button class="ctl" data-action="mode" aria-label="Switch between light and dark theme">'
        f'<span class="when-light">{ICON_MOON}Dark</span><span class="when-dark">{ICON_SUN}Light</span></button>'
        f'<button class="ctl" data-action="about" aria-haspopup="dialog">{ICON_INFO}About</button>'
        "</div></header>"
    )


def about_html():
    n_uns, n = len(UNSUPERVISED), len(MODELS)
    marks = "".join(tile(d) for d in range(10))
    return (
        '<div class="modal" id="about-modal" role="dialog" aria-modal="true" aria-labelledby="about-title">'
        '<div class="modal-card">'
        f'<button class="modal-x" data-action="close" aria-label="Close about dialog">{ICON_X}</button>'
        f'<div class="marks">{marks}</div>'
        '<h2 id="about-title">About NumNum ... Who is there?</h2>'
        f"<p>NumNum ... Who is there? compares how {n} machine learning algorithms read the same handwritten digit. "
        "Everything runs on your own computer.</p>"
        "<h3>How a drawing becomes an answer</h3>"
        "<ol>"
        "<li>Your drawing is cropped, scaled to fit 20 by 20 pixels and centered in a 28 by 28 frame, "
        "the way MNIST digits are prepared.</li>"
        "<li>Each selected model gives a probability for every digit from 0 to 9.</li>"
        "<li>The hard vote is the answer most models chose. The soft vote is the digit with the "
        "highest average probability.</li>"
        "</ol>"
        "<h3>Supervised and unsupervised models</h3>"
        f"<p>{n - n_uns} models learn from digits that come with labels. {n_uns} models never see a label "
        "while learning: they group the training images by pixel similarity, then each group is named "
        "after the digit most common inside it.</p>"
        "<h3>Data and tips</h3>"
        f"<p>The models were trained on {N_TRAIN:,} MNIST images and scored on its 10,000 test images "
        "(see the Leaderboard tab). They read best when the digit is large, centered and drawn in one clean stroke.</p>"
        '<p class="fine">Built with Python, scikit-learn and Gradio.</p>'
        "</div></div>"
    )


def algorithm_modal_html():
    return (
        '<div class="modal" id="algorithm-modal" role="dialog" aria-modal="true" aria-labelledby="algorithm-title">'
        '<div class="modal-card">'
        f'<button class="modal-x" data-action="close-algorithm" aria-label="Close algorithm explanation">{ICON_X}</button>'
        '<h2 id="algorithm-title">Algorithm</h2>'
        '<h3>Simple steps</h3><ol id="algorithm-steps"></ol>'
        '<h3>Example</h3><p id="algorithm-example"></p>'
        '<h3>More information</h3><p id="algorithm-info"></p>'
        '<h3>Why it helps</h3><p id="algorithm-strength"></p>'
        '<h3>When it can struggle</h3><p id="algorithm-limits"></p>'
        "</div></div>"
    )


STAGE_HEAD = (
    '<div class="stage-head"><h2>Draw a digit</h2>'
    "<p>Make it large and centered, in one clean stroke. Results update as you draw.</p></div>"
)

LIVE_IDLE = (
    '<div class="live idle"><span class="tile lg ghost">?</span><div><div class="live-title">Waiting for your drawing</div>'
    '<div class="live-sub">The answer shows up here as you draw.</div></div></div>'
)


def live_card(hard, soft, mean_probs):
    note = "" if hard == soft else f" The hard vote says {hard}."
    return (
        f'<div class="live">{tile(soft, "lg")}<div><div class="live-title">Most likely {soft}</div>'
        f'<div class="live-sub">{mean_probs[soft]:.0%} average confidence.{note}</div></div></div>'
    )


def result_card(hard, soft, counts, mean_probs, n_models):
    chips = "".join(
        f'<span class="chip" style="--c:{DIGIT_COLORS[d]}"><b>{d}</b> {c} vote{"s" if c != 1 else ""}</span>'
        for d, c in sorted(counts.items(), key=lambda t: -t[1])
    )
    bars = "".join(
        f'<div class="bar-row{" top" if d == soft else ""}"><span class="bar-label">{d}</span>'
        f'<span class="bar-track"><i style="width:{max(p * 100, 0.6):.1f}%;background:{DIGIT_COLORS[d]}"></i></span>'
        f'<span class="bar-val">{p:.0%}</span></div>'
        for d, p in enumerate(mean_probs)
    )
    tip = (
        '<div class="tip">Low confidence. Try drawing the digit larger, in one clean stroke.</div>'
        if mean_probs.max() < 0.5
        else ""
    )
    return (
        '<div class="result"><div class="verdicts">'
        f'<div class="verdict"><div class="label">Hard vote</div>{tile(hard, "xl")}'
        f'<div class="sub">Most common answer among {n_models} model{"s" if n_models != 1 else ""}</div></div>'
        f'<div class="verdict"><div class="label">Soft vote</div>{tile(soft, "xl")}'
        f'<div class="sub">Highest average probability, {mean_probs[soft]:.0%}</div></div>'
        f'<div class="verdict grow"><div class="label">Who voted for what</div><div class="chips">{chips}</div></div>'
        f'</div>{tip}<div class="label">Average probability per digit</div><div class="bars">{bars}</div></div>'
    )


def votes_card(rows, hard):
    html = ""
    for r in rows:
        d, c = r["pred"], r["conf"]
        cls = "vrow" if d == hard else "vrow dissent"
        html += (
            f'<div class="{cls}"><div class="vname">{r["name"]}{pill(r["name"])}</div>{tile(d)}'
            f'<span class="bar-track"><i style="width:{c * 100:.0f}%;background:{DIGIT_COLORS[d]}"></i></span>'
            f'<span class="bar-val">{c:.0%}</span></div>'
        )
    return f'<div class="sub wide">Highlighted rows disagree with the hard vote.</div><div class="votes">{html}</div>'


def hood_card(x, selected):
    shots = [(data_uri(x, 112, Image.NEAREST), "What the models see", "28 by 28 pixels")]
    for name in selected:
        if name in UNSUPERVISED:
            proto, cid, digit = MODELS[name].prototype(x.reshape(1, -1))
            shots.append((data_uri(proto, 112, Image.BICUBIC), name, f"Cluster {cid}, named {digit}"))
    inner = "".join(
        f'<figure class="shot"><img src="{u}" alt="{t}"><figcaption><b>{t}</b><span>{s}</span></figcaption></figure>'
        for u, t, s in shots
    )
    return (
        '<div class="label">Your drawing next to the average image of the cluster each unsupervised model matched</div>'
        f'<div class="shots">{inner}</div>'
    )


def make_map(x_flat=None):
    """Transparent PNG with mid-tone text, so one image works on both light and dark themes."""
    muted = "#8E86A8"
    fig = Figure(figsize=(6.6, 4.6), dpi=120)
    ax = fig.subplots()
    ax.set_facecolor("none")
    for d in range(10):
        m = MAP_Y == d
        ax.scatter(MAP_XY[m, 0], MAP_XY[m, 1], s=11, c=DIGIT_COLORS[d], alpha=0.65, linewidths=0, label=str(d))
    if x_flat is not None:
        px, py = PCA2.transform(x_flat)[0]
        ax.scatter([px], [py], s=420, marker="*", c="#FACC15", edgecolors="#2B2540", linewidths=1.6, zorder=5, label="You")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=muted)
    ax.grid(color=muted, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Where your digit lands on a 2D PCA map of 2,500 training digits",
                 loc="left", fontsize=10, color=muted, fontweight="bold")
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              markerscale=1.6, labelcolor=muted)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return f'<img class="map" src="{uri}" alt="2D map of training digits with your drawing marked by a star">'


@lru_cache(maxsize=1)
def base_map():
    return make_map(None)


def leaderboard_html(df):
    rows = ""
    for _, r in df.sort_values("Accuracy", ascending=False).iterrows():
        cls = {"Supervised": "sup", "Unsupervised": "uns", "Ensemble": "ens"}[r["Type"]]
        width = max(r["Accuracy"] - 0.5, 0) / 0.5 * 100
        name = r["Model"]
        rows += (
            '<div class="lrow"><div class="vname">'
            f'<div class="model-name"><b>{name}</b><button class="algo-info" data-algorithm="{name}" aria-label="Explain {name}">i</button></div>'
            f'<span class="pill {cls}">{r["Type"]}</span></div>'
            f'<span class="bar-track"><i class="{cls}" style="width:{width:.1f}%"></i></span>'
            f'<span class="bar-val strong">{r["Accuracy"]:.2%}</span></div>'
        )
    return (
        '<div class="sub wide">Accuracy on the 10,000 MNIST test images. Bars start at 50%.</div>'
        f'<div class="votes">{rows}</div>'
        '<div class="note"><b>How unsupervised models vote.</b> K-Means, Gaussian Mixture and Agglomerative (Ward) '
        "sort the training images into clusters using pixels only. Each cluster is then named after the digit "
        "most common inside it, and that name is the model's vote. Digit labels are never used to form the clusters.</div>"
    )


def predict(img, sup_sel, uns_sel):
    """Returns: live strip, prediction tab, votes tab, under-the-hood tab, digit map."""
    selected = [n for n in MODELS if n in (sup_sel or []) or n in (uns_sel or [])]
    x = preprocess(img)
    if x is None:
        wait = message_card("Nothing drawn yet", "Draw a digit from 0 to 9 on the canvas. Results update as you draw.")
        return LIVE_IDLE, wait, wait, wait, base_map()
    if not selected:
        none = message_card("No models selected", "Choose at least one model under the canvas.")
        return LIVE_IDLE, none, none, none, make_map(x.reshape(1, -1))

    rows, hard, soft, mean_probs, counts = vote(x.reshape(1, -1), selected)
    return (
        live_card(hard, soft, mean_probs),
        result_card(hard, soft, counts, mean_probs, len(selected)),
        votes_card(rows, hard),
        hood_card(x, selected),
        make_map(x.reshape(1, -1)),
    )


FONT = [gr.themes.GoogleFont("Nunito", weights=(400, 600, 700, 800)), "ui-rounded", "system-ui", "sans-serif"]
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=FONT,
    font_mono=FONT,
)

PALETTE_CSS = "".join(f'html[data-palette="{n}"]{{--accent:{c};}}\n' for n, c in PALETTES)

CSS = """
:root{
  --accent:#FF5C9E;
  --ink:#2B2540; --muted:#6F678A; --card:#ffffff; --card2:#FBF7FC;
  --teal-ink:#0B7F74; --violet-ink:#5B47D6; --warn-ink:#98570F;
  --peach:#FF9A5A;
  --accent-ink:color-mix(in srgb,var(--accent) 72%,#000);
  --accent-hover:color-mix(in srgb,var(--accent) 86%,#000);
  --bg:color-mix(in srgb,var(--accent) 6%,#fff);
  --accent-soft:color-mix(in srgb,var(--accent) 14%,var(--card));
  --line:color-mix(in srgb,var(--accent) 16%,var(--card));
  --track:color-mix(in srgb,var(--accent) 10%,var(--card));
  --shadow:color-mix(in srgb,var(--accent) 16%,transparent);
}
html[data-mode="dark"]{
  --ink:#F4EFFA; --muted:#A8A0C0; --card:#1D1A29; --card2:#262234;
  --teal-ink:#5EEAD4; --violet-ink:#C4B8FF; --warn-ink:#FFC999;
  --accent-ink:color-mix(in srgb,var(--accent) 55%,#fff);
  --bg:color-mix(in srgb,var(--accent) 9%,#0E0C15);
  --shadow:rgba(0,0,0,.35);
}
__PALETTES__

body,gradio-app{background:var(--bg)!important;}
.gradio-container{
  --body-background-fill:var(--bg); --body-text-color:var(--ink); --body-text-color-subdued:var(--muted);
  --background-fill-primary:var(--card); --background-fill-secondary:var(--card2);
  --block-background-fill:transparent; --block-border-width:0px; --block-shadow:none;
  --block-label-background-fill:var(--accent-soft); --block-label-text-color:var(--accent-ink);
  --block-title-text-color:var(--ink); --block-info-text-color:var(--muted);
  --border-color-primary:var(--line); --color-accent:var(--accent); --input-background-fill:var(--card);
  --button-primary-background-fill:var(--accent); --button-primary-background-fill-hover:var(--accent-hover);
  --button-primary-text-color:#fff; --button-primary-border-color:transparent;
  --button-primary-shadow:0 10px 22px var(--shadow);
  --button-secondary-background-fill:var(--card); --button-secondary-background-fill-hover:var(--accent-soft);
  --button-secondary-border-color:var(--line); --button-secondary-text-color:var(--ink);
  --checkbox-background-color:var(--card); --checkbox-border-color:var(--line);
  --checkbox-label-background-fill:var(--card); --checkbox-label-background-fill-hover:var(--accent-soft);
  --checkbox-label-background-fill-selected:var(--accent-soft);
  --checkbox-label-border-color:var(--line); --checkbox-label-border-color-selected:var(--accent);
  --checkbox-label-text-color:var(--ink); --checkbox-label-text-color-selected:var(--accent-ink);
  --checkbox-background-color-selected:var(--accent); --checkbox-border-color-selected:var(--accent);
  max-width:920px!important;margin:0 auto!important;background:transparent!important;color:var(--ink);
}
footer{display:none!important;}

.top{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:14px 24px;padding:8px 4px 16px;}
.brand h1{margin:0;font-size:32px;font-weight:800;letter-spacing:-.02em;line-height:1.05;color:var(--ink)!important;}
.brand p{margin:4px 0 0;font-size:14px;color:var(--muted);}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.swatches{display:flex;gap:6px;padding:6px; border-radius:14px;background:var(--card);border:1px solid var(--line);}
.sw{width:34px;height:34px;padding:0;border-radius:10px;border:0;background:var(--s);color:#fff;cursor:pointer;
  display:inline-flex;align-items:center;justify-content:center;font:800 16px 'Nunito',ui-rounded,system-ui,sans-serif;
  box-shadow:inset 0 -3px 0 rgba(0,0,0,.13);transition:transform .12s,box-shadow .12s;}
.sw:hover{transform:translateY(-1px);box-shadow:inset 0 -3px 0 rgba(0,0,0,.13),0 5px 12px color-mix(in srgb,var(--s) 28%,transparent);}
.sw[aria-pressed="true"]{box-shadow:inset 0 0 0 2px rgba(255,255,255,.8),inset 0 -3px 0 rgba(0,0,0,.13),0 0 0 2px var(--s);}
.ctl{display:inline-flex;align-items:center;gap:8px;height:38px;padding:0 15px;border-radius:999px;cursor:pointer;
  border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit;font-size:14px;font-weight:700;}
.ctl:hover{background:var(--accent-soft);}
.ctl svg{width:16px;height:16px;flex:none;}
.when-light,.when-dark{display:inline-flex;align-items:center;gap:8px;}
html[data-mode="dark"] .when-light,html:not([data-mode="dark"]) .when-dark{display:none;}
.sw:focus-visible,.ctl:focus-visible,.modal-x:focus-visible{outline:2px solid var(--accent);outline-offset:3px;}

.panel{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:28px!important;padding:22px!important;
  box-shadow:0 14px 34px var(--shadow);}
.stage-head h2{margin:0 0 4px;font-size:22px;font-weight:800;color:var(--ink)!important;}
.stage-head p{margin:0 0 4px;font-size:14px;color:var(--muted);}
.stage-canvas{border-radius:22px!important;overflow:hidden;border:2px dashed color-mix(in srgb,var(--accent) 50%,transparent)!important;}
.live{display:flex;align-items:center;gap:16px;padding:14px 18px;border-radius:20px;background:var(--accent-soft);}
.live.idle{background:var(--card2);}
.live-title{font-size:20px;font-weight:800;color:var(--ink);}
.live-sub{font-size:14px;color:var(--muted);}

.tile{display:inline-flex;align-items:center;justify-content:center;flex:none;color:#fff;font-weight:800;
  background:var(--c);box-shadow:inset 0 -4px 0 rgba(0,0,0,.14);}
.tile.sm{width:34px;height:34px;border-radius:11px;font-size:18px;}
.tile.lg{width:64px;height:64px;border-radius:20px;font-size:38px;box-shadow:inset 0 -5px 0 rgba(0,0,0,.14);}
.tile.xl{width:96px;height:96px;border-radius:30px;font-size:58px;box-shadow:inset 0 -7px 0 rgba(0,0,0,.14);}
.tile.ghost{background:var(--line);color:var(--muted);box-shadow:none;}

.label{font-size:14px;font-weight:700;color:var(--ink);}
.sub{font-size:13px;line-height:1.4;color:var(--muted);max-width:180px;}
.sub.wide{max-width:none;margin-bottom:10px;}

.verdicts{display:flex;flex-wrap:wrap;gap:18px 30px;align-items:flex-start;margin-bottom:24px;}
.verdict{display:flex;flex-direction:column;gap:8px;}
.verdict.grow{flex:1;min-width:170px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;}
.chip{display:inline-flex;gap:6px;align-items:center;padding:5px 12px;border-radius:999px;font-size:13px;font-weight:700;
  color:var(--c);background:color-mix(in srgb,var(--c) 14%,var(--card));}
.chip b{font-size:15px;}
.tip{margin:0 0 18px;padding:10px 14px;border-radius:12px;background:color-mix(in srgb,var(--peach) 16%,var(--card));
  color:var(--warn-ink);font-size:14px;font-weight:600;}

.bars{display:grid;gap:7px;margin-top:10px;}
.bar-row{display:grid;grid-template-columns:18px 1fr 44px;align-items:center;gap:12px;}
.bar-label{font-weight:800;color:var(--muted);}
.bar-row.top .bar-label{color:var(--ink);}
.bar-track{display:block;height:10px;border-radius:999px;background:var(--track);overflow:hidden;}
.bar-track i{display:block;height:100%;border-radius:999px;}
.bar-track i.sup{background:var(--accent);} .bar-track i.uns{background:#14B8A6;} .bar-track i.ens{background:#8B7CF6;}
.bar-val{font-size:13px;font-weight:700;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums;}
.bar-val.strong{color:var(--ink);}

.votes{display:grid;gap:8px;}
.vrow{display:grid;grid-template-columns:minmax(150px,1.3fr) 34px 1fr 44px;align-items:center;gap:14px;
  padding:10px 14px;border-radius:14px;background:var(--card2);border-left:4px solid transparent;}
.vrow.dissent{background:color-mix(in srgb,var(--peach) 14%,var(--card));border-left-color:var(--peach);}
.lrow{display:grid;grid-template-columns:minmax(190px,1.4fr) 1.4fr 64px;align-items:center;gap:14px;
  padding:10px 14px;border-radius:14px;background:var(--card2);}
.vname{display:flex;flex-direction:column;align-items:flex-start;gap:4px;font-weight:700;color:var(--ink);}
.model-name{display:flex;align-items:center;gap:7px;}
.model-name b{font-weight:800;}
.algo-info{width:20px;height:20px;padding:0;border-radius:50%;border:0;background:var(--accent);color:#fff;
  display:inline-flex;align-items:center;justify-content:center;font:800 13px Georgia,serif;cursor:pointer;
  box-shadow:0 3px 8px var(--shadow);transition:transform .12s;}
.algo-info:hover{transform:scale(1.12);}
.algo-info:focus-visible{outline:2px solid var(--accent-ink);outline-offset:2px;}
.pill{padding:2px 9px;border-radius:999px;font-size:12px;font-weight:700;}
.pill.sup{background:color-mix(in srgb,var(--accent) 16%,var(--card));color:var(--accent-ink);}
.pill.uns{background:color-mix(in srgb,#14B8A6 16%,var(--card));color:var(--teal-ink);}
.pill.ens{background:color-mix(in srgb,#8B7CF6 16%,var(--card));color:var(--violet-ink);}

.shots{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0 22px;}
.shot{margin:0;width:128px;}
.shot img{display:block;width:112px;height:112px;border-radius:18px;border:1px solid var(--line);background:#fff;}
.shot figcaption{margin-top:8px;display:flex;flex-direction:column;gap:2px;font-size:12px;color:var(--muted);}
.shot figcaption b{font-size:13px;color:var(--ink);}
.map{display:block;width:100%;height:auto;}

.empty{padding:44px 20px;text-align:center;border:2px dashed var(--line);border-radius:20px;}
.empty-title{font-size:18px;font-weight:800;margin-bottom:6px;color:var(--ink);}
.empty-text{font-size:14px;color:var(--muted);}
.note{margin-top:16px;padding:14px 16px;border-radius:14px;background:color-mix(in srgb,#14B8A6 11%,var(--card));
  color:var(--teal-ink);font-size:14px;line-height:1.55;}

button[role="tab"]{font-weight:700!important;}
button[role="tab"][aria-selected="true"]{color:var(--accent-ink)!important;}

.modal{position:fixed;inset:0;z-index:10000;display:none;align-items:center;justify-content:center;padding:20px;
  background:rgba(16,10,28,.55);backdrop-filter:blur(4px);font-family:'Nunito',ui-rounded,system-ui,sans-serif;}
.modal.open{display:flex;}
.modal-card{position:relative;width:min(620px,100%);max-height:88vh;overflow:auto;padding:28px;border-radius:28px;
  background:var(--card);color:var(--ink);border:1px solid var(--line);box-shadow:0 30px 80px rgba(0,0,0,.35);
  font-size:15px;line-height:1.55;}
.modal-card h2{margin:14px 0 8px;font-size:26px;font-weight:800;letter-spacing:-.01em;color:var(--ink);}
.modal-card h3{margin:20px 0 6px;font-size:16px;font-weight:800;color:var(--accent-ink);}
.modal-card p{margin:0 0 8px;color:var(--ink);}
.modal-card ol{margin:0;padding-left:22px;}
.modal-card li{margin:4px 0;}
.modal-card .fine{margin-top:18px;font-size:13px;color:var(--muted);}
.modal-card .marks{display:flex;gap:5px;flex-wrap:wrap;padding-right:44px;}
.modal-x{position:absolute;top:16px;right:16px;width:36px;height:36px;display:inline-flex;align-items:center;justify-content:center;
  border-radius:50%;border:1px solid var(--line);background:var(--card2);color:var(--ink);cursor:pointer;padding:0;}
.modal-x:hover{background:var(--accent-soft);}
.modal-x svg{width:16px;height:16px;}
""".replace("__PALETTES__", PALETTE_CSS)

JS = """
() => {
  const root = document.documentElement;
  const algorithmInfo = __ALGORITHM_INFO__;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
  };
  const setMode = (m) => {
    root.dataset.mode = m;
    document.body.classList.toggle('dark', m === 'dark');
    store.set('dl-mode', m);
  };
  const setPalette = (p) => {
    root.dataset.palette = p;
    document.querySelectorAll('.sw[data-palette]').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.palette === p)));
    store.set('dl-palette', p);
  };
  const modal = () => document.getElementById('about-modal');
  const openAbout = () => {
    const m = modal(); if (!m) return;
    m.classList.add('open');
    const x = m.querySelector('.modal-x'); if (x) x.focus();
  };
  const closeAbout = () => { const m = modal(); if (m) m.classList.remove('open'); };
  const algorithmModal = () => document.getElementById('algorithm-modal');
  const openAlgorithm = (name) => {
    const m = algorithmModal(); if (!m) return;
    const data = algorithmInfo[name] || {
      steps: ["Collect answers from the selected models.", "Combine those answers with one voting rule.", "Return the combined digit."],
      example: "Three models choose 7 and two models choose 1.",
      info: "An ensemble combines several algorithms to improve the overall decision.",
    strength: "Combining multiple viewpoints can reduce isolated mistakes.",
    limits: "The ensemble can inherit shared mistakes from every model in the group.",
    };
    m.querySelector('#algorithm-title').textContent = name;
    m.querySelector('#algorithm-example').textContent = data.example;
    m.querySelector('#algorithm-info').textContent = data.info;
    m.querySelector('#algorithm-strength').textContent = data.strength;
    m.querySelector('#algorithm-limits').textContent = data.limits;
    const steps = m.querySelector('#algorithm-steps');
    steps.replaceChildren();
    data.steps.forEach((step) => {
      const li = document.createElement('li');
      li.textContent = step;
      steps.appendChild(li);
    });
    m.classList.add('open');
    m.querySelector('.modal-x').focus();
  };
  const closeAlgorithm = () => { const m = algorithmModal(); if (m) m.classList.remove('open'); };

  const m = modal();
  if (m && m.parentElement !== document.body) document.body.appendChild(m);
  const am = algorithmModal();
  if (am && am.parentElement !== document.body) document.body.appendChild(am);

  const known = Array.from(document.querySelectorAll('.sw[data-palette]')).map((b) => b.dataset.palette);
  const savedPalette = store.get('dl-palette');
  const prefersDark = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches;
  setMode(store.get('dl-mode') || (prefersDark ? 'dark' : 'light'));
  setPalette(known.includes(savedPalette) ? savedPalette : 'bubblegum');

  if (window.__digitLabWired) return;
  window.__digitLabWired = true;
  document.addEventListener('click', (e) => {
    const sw = e.target.closest('.sw[data-palette]');
    if (sw) { setPalette(sw.dataset.palette); return; }
    const info = e.target.closest('.algo-info');
    if (info) { openAlgorithm(info.dataset.algorithm); return; }
    const act = e.target.closest('[data-action]');
    if (act) {
      const a = act.dataset.action;
      if (a === 'mode') setMode(root.dataset.mode === 'dark' ? 'light' : 'dark');
      else if (a === 'about') openAbout();
      else if (a === 'close') closeAbout();
      else if (a === 'close-algorithm') closeAlgorithm();
      return;
    }
    if (e.target.classList && e.target.classList.contains('modal')) {
      closeAbout();
      closeAlgorithm();
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeAbout(); closeAlgorithm(); }
  });
}
""".replace("__ALGORITHM_INFO__", json.dumps(ALGORITHM_INFO))


def supported(cls, **opts):
    """Keep only the keyword arguments this Gradio version understands."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in opts.items() if k in params}


STYLE = {"theme": THEME, "css": CSS}
STYLE_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters

with gr.Blocks(title="NumNum ... Who is there?", **({} if STYLE_IN_LAUNCH else STYLE)) as demo:
    gr.HTML(header_html())
    gr.HTML(about_html())
    gr.HTML(algorithm_modal_html())

    with gr.Column(elem_classes=["panel", "stage"]):
        gr.HTML(STAGE_HEAD)
        sketch = gr.Sketchpad(
            label="Drawing canvas",
            type="numpy",
            canvas_size=(560, 560),
            brush=gr.Brush(default_size=26, colors=["#000000"], color_mode="fixed"),
            elem_classes="stage-canvas",
            **supported(
                gr.Sketchpad,
                show_label=False,
                height=560,
                layers=False,
                show_download_button=False,
                show_fullscreen_button=False,
            ),
        )
        with gr.Row():
            btn = gr.Button("Predict", variant="primary")
            clear = gr.Button("Clear", variant="secondary")
        live = gr.HTML(LIVE_IDLE)

    with gr.Column(elem_classes="panel"):
        with gr.Accordion("Choose models", open=False):
            sup_box = gr.CheckboxGroup(
                choices=[m for m in MODELS if m not in UNSUPERVISED],
                value=[m for m in MODELS if m not in UNSUPERVISED],
                label="Supervised models",
                info="Trained with the digit labels.",
            )
            uns_box = gr.CheckboxGroup(
                choices=list(UNSUPERVISED),
                value=list(UNSUPERVISED),
                label="Unsupervised models",
                info="Cluster the images first, then name each cluster.",
            )

    with gr.Column(elem_classes="panel"):
        with gr.Tabs():
            with gr.Tab("Prediction"):
                result = gr.HTML(message_card("Nothing drawn yet", "Draw a digit from 0 to 9 on the canvas. Results update as you draw."))
            with gr.Tab("Model votes"):
                votes = gr.HTML(message_card("No votes yet", "Each model's answer and confidence appears here."))
            with gr.Tab("Under the hood"):
                hood = gr.HTML(message_card("Nothing to show yet", "Draw a digit to see what the models see."))
                map_out = gr.HTML(base_map())
            with gr.Tab("Leaderboard"):
                gr.HTML(leaderboard_html(ACCURACY))

    inputs = [sketch, sup_box, uns_box]
    outputs = [live, result, votes, hood, map_out]
    btn.click(predict, inputs, outputs)
    sketch.change(predict, inputs, outputs, trigger_mode="always_last", show_progress="hidden")
    sup_box.change(predict, inputs, outputs, show_progress="hidden")
    uns_box.change(predict, inputs, outputs, show_progress="hidden")
    clear.click(lambda: None, None, sketch)
    demo.load(None, None, None, js=JS)

if __name__ == "__main__":
    demo.launch(**(STYLE if STYLE_IN_LAUNCH else {}))