# Coding Standards for All Agents

1. Produce readable, human-understandable code.
2. Use clear names and small focused functions.
3. Keep modules cohesive; avoid needless abstraction.
4. Python must run on 3.10.11.
5. Add type hints to public functions and API models.
6. Use Pydantic for request/response validation.
7. Use `pathlib` for paths.
8. Use structured logging; do not log secrets or full user identifiers.
9. Use environment variables for configuration.
10. Never hardcode API keys.
11. No unnecessary comments, banners, decorative headers, or verbose docstrings.
12. Add a short comment only for non-obvious logic.
13. No fake data in production code.
14. No placeholder `pass`, TODO-only methods, or pretend implementations.
15. Avoid broad `except Exception` unless at an application boundary with logging and safe response.
16. Pin compatible dependency versions after testing.
17. Keep Render runtime memory low; avoid PyTorch and heavy ML stacks in production.
18. Do not download models or scrape websites during API startup.
19. Do not claim tests pass unless the command was executed and passed.
20. Before editing, inspect existing files and the current status.
21. Do not overwrite files owned by the other agent.
22. Update BUILD_STATUS.md after each completed work batch.
23. Keep generated explanations outside source code.
24. Never commit `.env`, tokens, cache directories, local venv, temporary crawls, or user data.
