# EstateIQ — Start Here

This starter pack is designed for `D:\EstateIQ`.

## 1. Install the project instructions

1. Extract the contents of this ZIP directly into `D:\EstateIQ`.
2. Preserve the folders and hidden `.github` folder.
3. Open `D:\EstateIQ` in VS Code.
4. Open `COPILOT_START_HERE.md` and follow its first prompt.
5. Open `CLAUDE_START_HERE.md`, upload the listed project documents to Claude.ai, and use the Claude master prompt.
6. Do not let Copilot and Claude edit the same files. Their ownership boundaries are defined in `docs/PROJECT_SPEC.md`.

## 2. Parallel work arrangement

- Copilot owns `scraper/`, `frontend/`, and the scraper CLI.
- Claude owns `app/` backend modules, API routes, RAG, retrieval, classification, LLM integration, and backend tests.
- Copilot can help integrate and fix issues after Claude's files are transferred.
- ChatGPT is the architecture and integration reviewer. Share generated files, test output, and `BUILD_STATUS.md` here for review.

Claude.ai in a browser does not automatically share your `D:\EstateIQ` filesystem. Download or copy its generated files into the exact paths it specifies.

## 3. Important time-saving rule

Do not ask either agent to regenerate the entire project after each change. Ask it to inspect the existing files, change only what is required, run the relevant tests, and update `BUILD_STATUS.md`.

## 4. Before running anything

- Python 3.10.11 is the required interpreter.
- Use a virtual environment.
- Never commit `.env`, API keys, tokens, or private data.
- Do not run OCR on every PDF. OCR is a fallback for scanned PDFs only.
- Do not scrape login-protected pages or bypass site restrictions.
- Use only public company information in the free NVIDIA OpenRouter endpoint.

## 5. Project completion definition

The project is complete only when:
- Official-source data has been collected and traceable URLs retained.
- The local FAISS index and SQLite metadata database build successfully.
- English and Arabic retrieval tests pass.
- The API and frontend work together.
- Unknown questions are safely declined rather than invented.
- Docker builds and runs locally.
- Render deployment is tested using the public URL.

A generated file is not evidence that its feature works. Tests are required.
