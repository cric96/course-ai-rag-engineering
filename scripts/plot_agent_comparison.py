"""Chart for the agent slide, from .cache/agent-comparison.json (compare_agent.py)."""

import json

import pandas as pd

from ragcourse.paths import ROOT
from ragcourse.viz import compare_chart

report = json.loads((ROOT / ".cache/agent-comparison.json").read_text())
runs = {"pipeline fissa": report["pipeline"], "agente": report["agent"]}
table = pd.DataFrame(
    {
        label: {
            "correttezza": run["correctness"],
            "risposte date": 1 - run["abstained"] / len(run["records"]),
        }
        for label, run in runs.items()
    }
)
fig = compare_chart(
    table,
    title=f"Agente RAG: correttezza {table.iloc[0, 0]:.2f} → {table.iloc[0, 1]:.2f}",
    fmt="{:.2f}",
)
fig.savefig(ROOT / "slides/assets/experiment-agent.svg")
