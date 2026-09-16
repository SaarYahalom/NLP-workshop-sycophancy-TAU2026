# Graphs

One matplotlib script per figure in the report. Each reads summary CSVs
from `../Results/{test}/…` and writes a PDF into this folder.

| Script | Reads from | Writes |
|---|---|---|
| `Answer_graph.py` | `../Results/answer/<variant>_llm_summary.csv` | `Answer_res_graph.pdf` |
| `Are_You_Sure_graph.py` | `../Results/are_you_sure/<variant>_summary.csv` | `AreYouSure_*.pdf` (results + method-validation) |
| `Are_You_Sure_Pushed_graph.py` | `../Results/are_you_sure_pushed/<variant>_summary.csv` | `AreYouSure_pushed_*.pdf` |
| `Feedback_graph.py` | `../Results/feedback/<variant>_llm_summary.csv` | `feedback_res_graph_*.pdf` |

Run from the repo root:

```powershell
python Graphs/Answer_graph.py
python Graphs/Are_You_Sure_graph.py
python Graphs/Are_You_Sure_Pushed_graph.py
python Graphs/Feedback_graph.py
```

Answer and feedback graphs consume the **LLM-regraded** summary CSVs (see
`Evaluators/README.md`); the two `are_you_sure*` graphs consume the raw
summary CSVs (no LLM regrade needed — pure letter-flip signal).
