(function () {
  "use strict";

  const form = document.getElementById("chat-form");
  const questionInput = document.getElementById("question");
  const companySelect = document.getElementById("company");
  const messages = document.getElementById("messages");
  const status = document.getElementById("status");
  const sendButton = document.getElementById("send-button");
  const clearButton = document.getElementById("clear-chat");

  function createTextElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    element.textContent = text;
    return element;
  }

  function safeHttpUrl(value) {
    if (typeof value !== "string") return null;
    try {
      const parsed = new URL(value, window.location.origin);
      return parsed.protocol === "https:" || parsed.protocol === "http:"
        ? parsed
        : null;
    } catch (_error) {
      return null;
    }
  }

  function appendMessage(role, text, rtl) {
    const article = document.createElement("article");
    article.className = "message " + role + (rtl ? " rtl" : "");
    article.appendChild(createTextElement("div", "message-content", text));
    messages.appendChild(article);
    messages.scrollTop = messages.scrollHeight;
    return article;
  }

  function appendCitation(content, source, number) {
    const parsed = source ? safeHttpUrl(source.url) : null;
    if (!parsed) {
      content.appendChild(document.createTextNode("[S" + number + "]"));
      return;
    }

    const link = document.createElement("a");
    link.className = "citation-link";
    link.href = parsed.href;
    link.textContent = "[" + number + "]";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = source.title || parsed.hostname;
    content.appendChild(link);
  }

  function appendAssistantAnswer(answer, sources, rtl) {
    const article = document.createElement("article");
    article.className = "message assistant" + (rtl ? " rtl" : "");

    const content = document.createElement("div");
    content.className = "message-content";

    // Supports [S1], [S1, S2], [S1;S2], and adjacent markers.
    const markerPattern = /\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]/gi;
    let cursor = 0;
    let match;

    while ((match = markerPattern.exec(answer)) !== null) {
      if (match.index > cursor) {
        content.appendChild(
          document.createTextNode(answer.slice(cursor, match.index))
        );
      }

      const ids = match[1].split(/[,;]/).map(function (value) {
        return Number(value.trim().replace(/^S/i, ""));
      });

      const seen = new Set();
      ids.forEach(function (number) {
        if (!Number.isInteger(number) || number < 1 || seen.has(number)) return;
        seen.add(number);
        const source = Array.isArray(sources) ? sources[number - 1] : null;
        appendCitation(content, source, number);
      });

      cursor = markerPattern.lastIndex;
    }

    if (cursor < answer.length) {
      content.appendChild(document.createTextNode(answer.slice(cursor)));
    }

    article.appendChild(content);
    messages.appendChild(article);
    messages.scrollTop = messages.scrollHeight;
    return article;
  }

  function appendSources(article, sources) {
    if (!Array.isArray(sources) || sources.length === 0) return;

    const container = document.createElement("div");
    container.className = "sources";
    container.appendChild(createTextElement("strong", "", "Sources"));

    sources.forEach(function (source, index) {
      if (!source) return;
      const parsed = safeHttpUrl(source.url);
      if (!parsed) return;

      const link = document.createElement("a");
      link.href = parsed.href;
      link.textContent = "[S" + (index + 1) + "] " +
        (source.title || parsed.hostname);
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      container.appendChild(link);
    });

    if (container.children.length > 1) article.appendChild(container);
  }

  async function sendQuestion(question) {
    const trimmed = question.trim();
    if (!trimmed || sendButton.disabled) return;

    appendMessage("user", trimmed, false);
    questionInput.value = "";
    status.className = "status";
    status.textContent = "Thinking...";
    sendButton.disabled = true;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          company: companySelect.value,
          language: "auto"
        })
      });

      let payload;
      try {
        payload = await response.json();
      } catch (_error) {
        throw new Error("The server returned an invalid response.");
      }

      if (!response.ok) {
        throw new Error(
          payload && payload.detail
            ? payload.detail
            : "The request could not be completed."
        );
      }

      const detectedLanguage = payload.detected_language || "en";
      const rtl = detectedLanguage === "ar";
      const answer = typeof payload.answer === "string" ? payload.answer : "";
      const sources = Array.isArray(payload.sources) ? payload.sources : [];

      const article = appendAssistantAnswer(
        answer || "No answer was returned.",
        sources,
        rtl
      );

      if (
        (rtl || detectedLanguage === "mixed") &&
        typeof payload.english_summary === "string" &&
        payload.english_summary.trim()
      ) {
        appendMessage(
          "assistant",
          "English summary: " + payload.english_summary,
          false
        );
      }

      appendSources(article, sources);
      status.textContent = payload.insufficient_evidence
        ? "The available evidence was insufficient."
        : "";
    } catch (error) {
      status.className = "status error";
      status.textContent = error instanceof Error
        ? error.message
        : "Unable to reach EstateIQ.";
    } finally {
      sendButton.disabled = false;
      questionInput.focus();
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    sendQuestion(questionInput.value);
  });

  document.querySelectorAll("[data-question]").forEach(function (button) {
    button.addEventListener("click", function () {
      questionInput.value = button.dataset.question || "";
      sendQuestion(questionInput.value);
    });
  });

  clearButton.addEventListener("click", function () {
    while (messages.firstChild) messages.removeChild(messages.firstChild);
    appendMessage(
      "assistant",
      "Ask a question about publicly available DarGlobal or Wasalt information.",
      false
    );
    status.className = "status";
    status.textContent = "";
  });
})();
