"""Project the baseline index and an example query onto a two-dimensional map."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from qdrant_client import QdrantClient

from ragcourse.cache import cache_embeddings
from ragcourse.models import embedding_model
from ragcourse.paths import ROOT
from ragcourse.viz import ORANGE, STYLE, TEAL

index = ROOT / ".cache" / "qdrant" / "e5-small-1000"
question = "Posso comprare un computer con il contributo?"

client = QdrantClient(path=str(index))
try:
    vectors = []
    offset = None
    while True:
        points, offset = client.scroll(
            "e5-small-1000",
            offset=offset,
            limit=512,
            with_vectors=True,
            with_payload=False,
        )
        vectors.extend(point.vector for point in points)
        if offset is None:
            break
finally:
    client.close()

matrix = np.asarray(vectors, dtype=np.float32)
center = matrix.mean(axis=0)
centered = matrix - center
_, axes = np.linalg.eigh(centered.T @ centered)
projection = axes[:, -2:][:, ::-1]
points = centered @ projection
cache_embeddings()
query = np.asarray(embedding_model("e5-small").embed_query(question))
query_point = (query - center) @ projection

with plt.rc_context(STYLE):
    fig = Figure(figsize=(10, 4.5))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.78)
    ax = fig.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=6, c=TEAL, alpha=0.16, rasterized=True)
    ax.scatter(*query_point, s=300, c=ORANGE, marker="*", zorder=3)
    ax.annotate(
        "Domanda",
        xy=query_point,
        xytext=(0.80, 1.20),
        textcoords="axes fraction",
        ha="center",
        color=TEAL,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "shrinkB": 12},
        annotation_clip=False,
    )
    ax.set_xlabel("Prima componente")
    ax.set_ylabel("Seconda componente")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    output = ROOT / "slides" / "assets" / "corpus-embeddings.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
print(f"Saved {len(matrix)} vectors to {output}")
