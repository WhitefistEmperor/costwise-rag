---
title: Costwise RAG
emoji: 💸
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Live demo of [Costwise RAG](https://github.com/WhitefistEmperor/costwise-rag): a RAG service with a
verified semantic cache and a cost-aware model router. This demo uses a mock model that echoes the
retrieved context, so answers are not real LLM output; the cache, router and dashboard are real.
Ask a question, then ask it again in different words and watch the cache hit.
