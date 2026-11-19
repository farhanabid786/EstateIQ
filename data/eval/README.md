# Retrieval Evaluation

`questions.jsonl` contains 11 deterministic questions grounded in the 40-record DarGlobal dataset.
Nine questions are supported by official source documents. Two records are marked `out_of_scope`.

Run:

```powershell
py -3.10 -m scripts.evaluate_retrieval --questions data/eval/questions.jsonl --mode hybrid
py -3.10 -m scripts.evaluate_retrieval --questions data/eval/questions.jsonl --mode dense
py -3.10 -m scripts.evaluate_retrieval --questions data/eval/questions.jsonl --mode bm25
```

The Arabic questions test cross-language retrieval against English source evidence.
No result claims Arabic source coverage because the dataset contains no Arabic documents.

`correct_company_attribution` excludes empty retrievals. Its denominator is the number
of non-out-of-scope, non-empty evaluated retrievals. A `darglobal` or `wasalt` scope
requires every returned document ID to use that company prefix. A `both` scope accepts
DarGlobal and Wasalt document prefixes and rejects unknown or foreign prefixes.

The report separates retrieval language capability by capability name:
English-to-English evidence, Arabic-to-English evidence, unsupported queries and
unsupported Arabic queries. Arabic-to-English metrics do not claim original Arabic
source coverage.
