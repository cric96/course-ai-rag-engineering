"""Generate the embedding projection used in the slides."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ragcourse.models import embedding_model
from ragcourse.paths import ROOT
from ragcourse.viz import ORANGE, STYLE, TEAL

question = "Posso comprare un computer con il contributo?"
passages = [
    "Le spese per l'acquisto di hardware sono ammissibili.",
    "Il contributo è concesso a fondo perduto.",
    "La domanda va presentata entro il 2 novembre 2026.",
]
embeddings = embedding_model("e5-small")
vectors = np.array(
    [embeddings.embed_query(question), *embeddings.embed_documents(passages)]
)
centered = vectors - vectors.mean(axis=0)
_, _, axes = np.linalg.svd(centered, full_matrices=False)
points = centered @ axes[:2].T

with plt.rc_context(STYLE):
    fig = Figure(figsize=(10, 4.2), layout="constrained")
    ax = fig.subplots()
    ax.scatter(points[1:, 0], points[1:, 1], s=180, c=TEAL)
    ax.scatter(points[0, 0], points[0, 1], s=220, c=ORANGE, marker="*")
    for point, label in zip(
        points, ["Domanda", "Hardware", "Contributo", "Scadenza"], strict=True
    ):
        ax.annotate(label, point, xytext=(10, 10), textcoords="offset points")
    ax.margins(0.35)
    ax.set_xlabel("Prima componente")
    ax.set_ylabel("Seconda componente")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    output = ROOT / "slides" / "assets" / "embedding-example.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
print(f"Saved {output}")
