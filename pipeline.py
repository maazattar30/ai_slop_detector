"""
app.py — Gradio interface for HF Space
Password protected AI video detector
"""

import gradio as gr
import json
import os
import base64
from PIL import Image

from config import PASSWORD, BUCKETS, BUCKET_THRESHOLDS
from pipeline import run_pipeline


# ─────────────────────────────────────────────
# PASSWORD CHECK
# ─────────────────────────────────────────────

def check_password(password: str) -> bool:
    if not PASSWORD:
        return True   # no password set = open access
    return password.strip() == PASSWORD.strip()


# ─────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────

def analyze(
    url:         str,
    title:       str,
    description: str,
    password:    str,
    progress=gr.Progress(track_tqdm=True)
) -> tuple:
    """
    Main Gradio function.
    Returns (status_text, result_json, report_html)
    """
    # Password check
    if not check_password(password):
        return (
            "❌ Incorrect password",
            "{}",
            "<p style='color:red'>Access denied</p>"
        )

    # URL validation
    url = url.strip()
    if not url or "youtube" not in url and "youtu.be" not in url:
        return (
            "❌ Please enter a valid YouTube URL",
            "{}",
            "<p style='color:red'>Invalid URL</p>"
        )

    def progress_cb(pct, msg):
        progress(pct, desc=msg)

    try:
        progress(0, desc="Starting pipeline...")
        result = run_pipeline(
            url         = url,
            title       = title,
            description = description,
            progress_cb = progress_cb,
        )

        if result.get("error"):
            return (
                f"❌ Error: {result['error']}",
                json.dumps(result, indent=2),
                f"<p style='color:red'>Error: {result['error']}</p>"
            )

        html   = _build_report_html(result)
        status = (
            f"✅ Done — "
            f"{result['bucket_name']} "
            f"(Score: {result['final_score']}/100)"
        )

        return (
            status,
            json.dumps(result, indent=2, default=str),
            html
        )

    except Exception as e:
        return (
            f"❌ Unexpected error: {str(e)}",
            "{}",
            f"<p style='color:red'>{str(e)}</p>"
        )


# ─────────────────────────────────────────────
# HTML REPORT BUILDER
# ─────────────────────────────────────────────

def _build_report_html(result: dict) -> str:
    bucket      = result["bucket"]
    b_info      = BUCKETS[bucket]
    final_score = result["final_score"]
    mod_score   = result["module_score"]
    llm_score   = result["llm_score"]
    llm         = result.get("llm_result", {})
    confidence  = result.get("confidence", 0.5)
    color       = b_info["color"]
    emoji       = b_info["emoji"]
    name        = b_info["name"]

    def score_bar(score, color):
        pct = min(max(score, 0), 100)
        return (
            f"<div style='background:#eee;border-radius:6px;"
            f"height:10px;margin:6px 0'>"
            f"<div style='background:{color};width:{pct}%;"
            f"height:10px;border-radius:6px'></div></div>"
        )

    def pill(label):
        colors = {
            "STRONG AI":    ("background:#FEE2E2", "color:#DC2626"),
            "MODERATE AI":  ("background:#FEF3C7", "color:#D97706"),
            "NEUTRAL":      ("background:#F1F5F9", "color:#64748B"),
            "MODERATE REAL":("background:#DBEAFE", "color:#2563EB"),
            "STRONG REAL":  ("background:#DCFCE7", "color:#16A34A"),
        }
        bg, fg = colors.get(
            label, ("background:#F1F5F9", "color:#94A3B8")
        )
        return (
            f"<span style='{bg};{fg};padding:2px 8px;"
            f"border-radius:99px;font-size:11px;"
            f"font-weight:700'>{label}</span>"
        )

    # Signal table rows
    sig_rows = result.get("signal_rows", [])
    table_rows = ""
    for row in sig_rows:
        table_rows += (
            f"<tr>"
            f"<td style='padding:8px 12px;font-weight:600'>"
            f"{row['feature']}</td>"
            f"<td style='padding:8px 12px;font-family:monospace'>"
            f"{row['value']}</td>"
            f"<td style='padding:8px 12px'>{pill(row['label'])}</td>"
            f"<td style='padding:8px 12px;color:#64748B;font-size:12px'>"
            f"{row['ranges']}</td>"
            f"<td style='padding:8px 12px;color:#64748B;font-size:12px'>"
            f"{row['definition']}</td>"
            f"</tr>"
        )

    # LLM evidence lists
    def ul(items, color="#334155"):
        if not items:
            return "<li style='color:#94A3B8'>None</li>"
        return "".join(
            f"<li style='margin:4px 0;color:{color}'>{i}</li>"
            for i in items
        )

    prim_html  = ul(llm.get("primary_signals", []))
    vis_html   = ul(llm.get("visual_evidence", []))
    flags_html = ul(llm.get("red_flags", []), "#DC2626")

    override_html = ""
    if llm.get("module_override"):
        override_html = (
            f"<div style='background:#FEF9E7;border-left:4px solid"
            f" #F39C12;padding:12px;margin:12px 0;border-radius:4px'>"
            f"<b>⚠️ LLM override:</b> {llm.get('override_reason','')}"
            f"</div>"
        )

    html = f"""
<div style='font-family:Arial,sans-serif;max-width:1000px;
            margin:0 auto;padding:20px'>

  <!-- Header -->
  <div style='background:linear-gradient(135deg,#1F4E79,#2E75B6);
              color:white;padding:20px;border-radius:10px;
              margin-bottom:16px'>
    <h2 style='margin:0'>🎬 MiQ AI Content Detector</h2>
    <p style='margin:4px 0 0 0;opacity:0.85;font-size:13px'>
      {result.get('title','')[:80]}
    </p>
  </div>

  <!-- Verdict -->
  <div style='background:{color}15;border:2px solid {color};
              border-radius:10px;padding:20px;margin-bottom:16px;
              text-align:center'>
    <div style='font-size:42px'>{emoji}</div>
    <div style='font-size:26px;font-weight:900;color:{color}'>
      Bucket {bucket} — {name}
    </div>
    <div style='color:#566573;margin-top:6px'>
      AI Score: <b style='color:{color}'>{final_score}</b>/100
      &nbsp;|&nbsp;
      Confidence: <b>{confidence:.0%}</b>
    </div>
  </div>

  <!-- Score cards -->
  <div style='display:flex;gap:12px;margin-bottom:16px'>
    <div style='flex:1;background:white;border:1px solid #ddd;
                border-radius:8px;padding:14px'>
      <div style='font-size:11px;color:#94A3B8'>MODULE SCORE</div>
      <div style='font-size:28px;font-weight:700;color:#2563EB'>
        {mod_score:.1f}
      </div>
      {score_bar(mod_score, "#2563EB")}
    </div>
    <div style='flex:1;background:white;border:1px solid #ddd;
                border-radius:8px;padding:14px'>
      <div style='font-size:11px;color:#94A3B8'>LLM SCORE</div>
      <div style='font-size:28px;font-weight:700;color:#7C3AED'>
        {llm_score}
      </div>
      {score_bar(llm_score, "#7C3AED")}
    </div>
    <div style='flex:1;background:{color}15;
                border:2px solid {color};
                border-radius:8px;padding:14px'>
      <div style='font-size:11px;color:#94A3B8'>FINAL SCORE</div>
      <div style='font-size:28px;font-weight:700;color:{color}'>
        {final_score}
      </div>
      {score_bar(final_score, color)}
    </div>
  </div>

  {override_html}

  <!-- Explanation -->
  <div style='background:#EBF5FB;border-left:4px solid #2E75B6;
              padding:14px;border-radius:4px;margin-bottom:16px'>
    <b style='color:#1F4E79'>📋 Explanation:</b><br>
    <p style='margin:8px 0 0 0;color:#2C3E50'>
      {llm.get('explanation','')}
    </p>
  </div>

  <!-- Evidence panels -->
  <div style='display:flex;gap:12px;margin-bottom:16px'>
    <div style='flex:1;background:white;border:1px solid #ddd;
                border-radius:8px;padding:14px'>
      <b style='color:#1F4E79'>🎯 Primary Signals</b>
      <ul style='margin:8px 0 0 0;padding-left:18px'>
        {prim_html}
      </ul>
    </div>
    <div style='flex:1;background:white;border:1px solid #ddd;
                border-radius:8px;padding:14px'>
      <b style='color:#1F4E79'>👁 Visual Evidence</b>
      <ul style='margin:8px 0 0 0;padding-left:18px'>
        {vis_html}
      </ul>
    </div>
    <div style='flex:1;background:white;border:1px solid #ddd;
                border-radius:8px;padding:14px'>
      <b style='color:#DC2626'>🚩 AI Red Flags</b>
      <ul style='margin:8px 0 0 0;padding-left:18px'>
        {flags_html}
      </ul>
    </div>
  </div>

  <!-- Signal table -->
  <div style='background:white;border:1px solid #ddd;
              border-radius:8px;margin-bottom:16px'>
    <div style='padding:14px;border-bottom:1px solid #eee'>
      <b style='color:#1F4E79'>📊 Signal Breakdown</b>
    </div>
    <div style='overflow-x:auto'>
      <table style='width:100%;border-collapse:collapse;
                    font-size:13px'>
        <thead>
          <tr style='background:#1F4E79;color:white'>
            <th style='padding:10px 12px;text-align:left'>Feature</th>
            <th style='padding:10px 12px;text-align:left'>Value</th>
            <th style='padding:10px 12px;text-align:left'>Signal</th>
            <th style='padding:10px 12px;text-align:left'>Ranges</th>
            <th style='padding:10px 12px;text-align:left'>
              What it measures
            </th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </div>

  <div style='color:#94A3B8;font-size:11px;text-align:center'>
    MiQ AI Content Detector V1 &nbsp;|&nbsp;
    LLM: Llama 4 Scout (Groq) &nbsp;|&nbsp;
    SigLIP: {"enabled" if result.get("signal_rows") else "CPU mode"}
  </div>

</div>"""

    return html


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="AI Content Detector",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🎬 MiQ AI Content Detector
    Paste a YouTube URL to classify it as Human, Human+AI Tools,
    or AI Generated using 10 signal modules + Llama 4 Scout.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            url_input   = gr.Textbox(
                label="YouTube URL *",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            title_input = gr.Textbox(
                label="Video Title (optional — auto-fetched)",
                placeholder="e.g. Life at MiQ"
            )
            desc_input  = gr.Textbox(
                label="Description (optional)",
                placeholder="Paste description for text signal analysis...",
                lines=3
            )
            pass_input  = gr.Textbox(
                label="Password",
                type="password",
                placeholder="Enter access password"
            )
            run_btn     = gr.Button(
                "▶  Analyze Video",
                variant="primary"
            )

        with gr.Column(scale=1):
            gr.Markdown("""
            **Bucket Reference**

            🟢 **B1 Human Only**
            Real camera, no AI

            🔵 **B2 Human + AI Tools**
            Human-made, AI for editing.
            Also used when uncertain.

            🔴 **B3 AI Generated**
            Text-to-video, deepfake, TTS

            ---
            **Processing time:** ~2-4 min

            **Signals:** 10 modules

            **LLM:** Llama 4 Scout (Groq)
            """)

    status_out = gr.Textbox(label="Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("📊 Report Card"):
            report_html = gr.HTML()
        with gr.Tab("🔧 Raw JSON"):
            json_out = gr.Code(language="json")

    run_btn.click(
        fn=analyze,
        inputs=[
            url_input, title_input,
            desc_input, pass_input
        ],
        outputs=[status_out, json_out, report_html]
    )

if __name__ == "_`_main__":
    demo.launch()