---
title: AI Slop Detector
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# AI Slop Detector V1

Classifies YouTube videos into 3 buckets using 10 signal
modules + Llama 4 Scout multimodal reasoning.

## Buckets
- 🟢 B1 Human Only
- 🔵 B2 Human + AI Tools  
- 🔴 B3 AI Generated

## Signals
Audio, texture, motion, optical flow, metadata,
text, evidence builder, LLM judge.