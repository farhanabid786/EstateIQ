# Model and Hosting Decisions

Research date: 2026-11-18. Recheck availability before deployment.

## LLM candidate
OpenRouter:
`nvidia/nemotron-3.5-super-vl:free`

Model page:
https://openrouter.ai/nvidia/nemotron-3.5-super-vl:free

OpenRouter lists this variant as free with a 1M-token context and structured-output/tool support. Free availability, rate limits, provider routing and policies may change. The free NVIDIA endpoint's page states that requests may be logged and used to improve NVIDIA products. Use public company-source context only. Do not send personal data, secrets, HR emails, or confidential information.

This is the selected initial candidate, not a proven universal benchmark winner. Run a real smoke test and the project's own bilingual RAG evaluation. Keep `OPENROUTER_MODEL` configurable.

## Embedding candidate
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via FastEmbed/ONNX Runtime.

Reason:
- Multilingual semantic retrieval.
- FastEmbed has a supported model entry.
- ONNX avoids a full PyTorch runtime.
- More appropriate for a 512 MB Render Free service than a heavier PyTorch model.

Verify exact model identifier, vector dimension and model support against the pinned FastEmbed version. Use its model-specific input formatting; do not apply E5 prefixes unless the model documentation says to.

Alternative quality candidate: `intfloat/multilingual-e5-small`, but custom ONNX/FastEmbed setup is extra work and is not the default under the deadline.

## Hosting constraint
Render pricing currently lists Free web service at 512 MB RAM. Free service filesystem is ephemeral and the service spins down after inactivity. Package the read-only index/database into the image and keep runtime memory low. Do not assume local writes survive restarts.
