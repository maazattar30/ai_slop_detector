"""
app.py — Gradio interface
Password protected AI video detector
"""

import gradio as gr
import json
import os
import base64
from PIL import Image

from config import PASSWORD, BUCKETS
from pipeline import run_pipeline


# ─────────────────────────────────────────────
# PASSWORD CHECK
# ─────────────────────────────────────────────

def check_password(password: str) -> bool:
    if not PASSWORD:
        return True
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
        return gr.Error(
            "❌ Incorrect password",
            "{}",
            "<p style='color:red;padding:20px'>Access denied — wrong password</p>"
        )

    # URL validation
    url = url.strip()
    if not url:
        return (
            "❌ Please enter a YouTube URL",
            "{}",
            "<p style='color:red;padding:20px'>No URL provided</p>"
        )

    if "youtube" not in url and "youtu.be" not in url:
        return (
            "❌ Please enter a valid YouTube URL",
            "{}",
            "<p style='color:red;padding:20px'>Invalid URL — must be YouTube</p>"
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
                f"<p style='color:red;padding:20px'>Error: {result['error']}</p>"
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
            f"<p style='color:red;padding:20px'>{str(e)}</p>"
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

    def score_bar(score, bar_color):
        pct = min(max(score, 0), 100)
        return (
            f"<div style='background:#eee;border-radius:6px;"
            f"height:10px;margin:6px 0'>"
            f"<div style='background:{bar_color};width:{pct}%;"
            f"height:10px;border-radius:6px'></div></div>"
        )

    def pill(label):
        colors = {
            "STRONG AI":     ("background:#FEE2E2", "color:#DC2626"),
            "MODERATE AI":   ("background:#FEF3C7", "color:#D97706"),
            "NEUTRAL":       ("background:#F1F5F9", "color:#64748B"),
            "MODERATE REAL": ("background:#DBEAFE", "color:#2563EB"),
            "STRONG REAL":   ("background:#DCFCE7", "color:#16A34A"),
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
    sig_rows   = result.get("signal_rows", [])
    table_rows = ""
    for row in sig_rows:
        table_rows += (
            f"<tr style='border-bottom:1px solid #F1F5F9'>"
            f"<td style='padding:8px 12px;font-weight:600;color:#2C3E50'>"
            f"{row['feature']}</td>"
            f"<td style='padding:8px 12px;font-family:monospace;color:#17202A'>"
            f"{row['value']}</td>"
            f"<td style='padding:8px 12px'>{pill(row['label'])}</td>"
            f"<td style='padding:8px 12px;color:#64748B;font-size:12px'>"
            f"{row.get('ranges','')}</td>"
            f"<td style='padding:8px 12px;color:#64748B;font-size:12px'>"
            f"{row.get('definition','')}</td>"
            f"</tr>"
        )

    # Evidence lists
    def ul(items, item_color="#334155"):
        if not items:
            return "<li style='color:#94A3B8'>None detected</li>"
        return "".join(
            f"<li style='margin:4px 0;color:{item_color}'>{i}</li>"
            for i in items
        )

    prim_html  = ul(llm.get("primary_signals", []))
    vis_html   = ul(llm.get("visual_evidence", []))
    flags_html = ul(llm.get("red_flags", []), "#DC2626")

    override_html = ""
    if llm.get("module_override"):
        override_html = (
            f"<div style='background:#FEF9E7;border-left:4px solid "
            f"#F39C12;padding:12px;margin:12px 0;border-radius:4px'>"
            f"<b>⚠️ LLM override:</b> "
            f"{llm.get('override_reason', '')}</div>"
        )

    html = f"""
<div style='font-family:Arial,sans-serif;max-width:1000px;
            margin:0 auto;padding:20px'>

  <!-- Header -->
  <div style='background:linear-gradient(135deg,#0F172A,#1E3A5F);
              color:white;padding:22px;border-radius:12px;
              margin-bottom:18px'>
    <h2 style='margin:0;font-size:22px'>🎬 AI Slop Detector</h2>
    <p style='margin:4px 0 0 0;opacity:0.8;font-size:13px'>
      {result.get('title', '')[:80]}
    </p>
  </div>

  <!-- Verdict card -->
  <div style='background:{color}15;border:2px solid {color};
              border-radius:12px;padding:24px;margin-bottom:18px;
              text-align:center'>
    <div style='font-size:48px'>{emoji}</div>
    <div style='font-size:28px;font-weight:900;color:{color};
                margin:6px 0'>
      Bucket {bucket} — {name}
    </div>
    <div style='color:#566573'>
      AI Score: <b style='color:{color}'>{final_score}</b>/100
      &nbsp;|&nbsp;
      Confidence: <b style='color:{color}'>{confidence:.0%}</b>
    </div>
  </div>

  <!-- Score cards -->
  <div style='display:flex;gap:14px;margin-bottom:18px'>
    <div style='flex:1;background:white;border:1px solid #E2E8F0;
                border-radius:8px;padding:16px'>
      <div style='font-size:11px;color:#94A3B8;
                  text-transform:uppercase;letter-spacing:0.1em'>
        Module Score
      </div>
      <div style='font-size:30px;font-weight:700;color:#2563EB'>
        {mod_score:.1f}
      </div>
      {score_bar(mod_score, "#2563EB")}
      <div style='font-size:11px;color:#94A3B8'>
        12 signals, Cohen d-weighted
      </div>
    </div>
    <div style='flex:1;background:white;border:1px solid #E2E8F0;
                border-radius:8px;padding:16px'>
      <div style='font-size:11px;color:#94A3B8;
                  text-transform:uppercase;letter-spacing:0.1em'>
        LLM Score
      </div>
      <div style='font-size:30px;font-weight:700;color:#7C3AED'>
        {llm_score}
      </div>
      {score_bar(llm_score, "#7C3AED")}
      <div style='font-size:11px;color:#94A3B8'>
        Llama 4 Scout visual reasoning
      </div>
    </div>
    <div style='flex:1;background:{color}10;
                border:2px solid {color};
                border-radius:8px;padding:16px'>
      <div style='font-size:11px;color:#94A3B8;
                  text-transform:uppercase;letter-spacing:0.1em'>
        Final Score
      </div>
      <div style='font-size:30px;font-weight:700;color:{color}'>
        {final_score}
      </div>
      {score_bar(final_score, color)}
      <div style='font-size:11px;color:#94A3B8'>
        55% module + 45% LLM
      </div>
    </div>
  </div>

  {override_html}

  <!-- Explanation -->
  <div style='background:#EBF5FB;border-left:4px solid #2563EB;
              padding:16px;border-radius:4px;margin-bottom:18px'>
    <b style='color:#1F4E79'>📋 Explanation:</b><br>
    <p style='margin:8px 0 0 0;color:#2C3E50;line-height:1.7'>
      {llm.get('explanation', 'No explanation available.')}
    </p>
  </div>

  <!-- Evidence panels -->
  <div style='display:flex;gap:14px;margin-bottom:18px'>
    <div style='flex:1;background:white;border:1px solid #E2E8F0;
                border-radius:8px;padding:14px'>
      <b style='color:#1F4E79'>🎯 Primary Signals</b>
      <ul style='margin:8px 0 0 0;padding-left:18px'>
        {prim_html}
      </ul>
    </div>
    <div style='flex:1;background:white;border:1px solid #E2E8F0;
                border-radius:8px;padding:14px'>
      <b style='color:#1F4E79'>👁 Visual Evidence</b>
      <ul style='margin:8px 0 0 0;padding-left:18px'>
        {vis_html}
      </ul>
    </div>
    <div style='flex:1;background:white;border:1px solid #E2E8F0;
                border-radius:8px;padding:14px'>
      <b style='color:#DC2626'>🚩 AI Red Flags</b>
      <ul style='margin:8px 0 0 0;padding-left:18px'>
        {flags_html}
      </ul>
    </div>
  </div>

  <!-- Signal table -->
  <div style='background:white;border:1px solid #E2E8F0;
              border-radius:8px;margin-bottom:18px'>
    <div style='padding:14px;border-bottom:1px solid #F1F5F9'>
      <b style='color:#1F4E79;font-size:15px'>
        📊 Signal Breakdown
      </b>
    </div>
    <div style='overflow-x:auto'>
      <table style='width:100%;border-collapse:collapse;
                    font-size:13px'>
        <thead>
          <tr style='background:#1F4E79;color:white'>
            <th style='padding:10px 12px;text-align:left'>
              Feature
            </th>
            <th style='padding:10px 12px;text-align:left'>
              Value
            </th>
            <th style='padding:10px 12px;text-align:left'>
              Signal
            </th>
            <th style='padding:10px 12px;text-align:left'>
              Ranges
            </th>
            <th style='padding:10px 12px;text-align:left'>
              What it measures
            </th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Footer -->
  <div style='color:#94A3B8;font-size:11px;text-align:center;
              padding:8px'>
    AI Slop Detector V1 &nbsp;|&nbsp;
    Signals: 10 modules &nbsp;|&nbsp;
    LLM: Llama 4 Scout (Groq)
  </div>

</div>"""

    return html


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="AI Slop Detector",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🎬 AI Slop Detector
    Paste a YouTube URL to classify it as **Human Only**,
    **Human + AI Tools**, or **AI Generated** using
    10 signal modules + Llama 4 Scout visual reasoning.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            url_input   = gr.Textbox(
                label       = "YouTube URL *",
                placeholder = "https://www.youtube.com/watch?v=..."
            )
            title_input = gr.Textbox(
                label       = "Video Title (optional — auto-fetched)",
                placeholder = "e.g. My Travel Vlog"
            )
            desc_input  = gr.Textbox(
                label       = "Description (optional)",
                placeholder = "Paste description for text signal analysis...",
                lines       = 3
            )
            pass_input  = gr.Textbox(
                label       = "Password",
                type        = "password",
                placeholder = "Enter access password"
            )
            run_btn = gr.Button(
                "▶  Analyze Video",
                variant = "primary",
                size    = "lg"
            )

        with gr.Column(scale=1):
            gr.Markdown("""
            ### Bucket Reference

            🟢 **B1 — Human Only**
            Real camera, real people, no AI

            🔵 **B2 — Human + AI Tools**
            Human-made with AI assistance.
            Also used when uncertain.

            🔴 **B3 — AI Generated**
            Text-to-video, deepfake, TTS, AI avatar

            ---
            ⏱ **Processing time:** ~2-4 min

            🔬 **Signals:** 10 modules

            🤖 **LLM:** Llama 4 Scout (Groq)
            """)

    status_out = gr.Textbox(
        label       = "Status",
        interactive = False
    )

    with gr.Tabs():
        with gr.Tab("📊 Report Card"):
            report_html = gr.HTML()
        with gr.Tab("🔧 Raw JSON"):
            json_out = gr.Code(language="json")

    run_btn.click(
        fn      = analyze,
        inputs  = [url_input, title_input, desc_input, pass_input],
        outputs = [status_out, json_out, report_html]
    )

if __name__ == "__main__":
    # Railway injects PORT env var; fall back to 7860
    port = int(os.environ.get("PORT", 7860))

    # Add this right before demo.launch() in app.py
    import subprocess
    def test_ytdlp():
        result = subprocess.run(
            ["yt-dlp", "--extractor-args", "youtube:player_client=tv_embedded",
            "--dump-json", "--no-download", 
            "https://youtu.be/WfbRYr5Xm-M"],
            capture_output=True, text=True
        )
        print("STDOUT:", result.stdout[:200])
        print("STDERR:", result.stderr[:500])
        print("RETURNCODE:", result.returncode)

    test_ytdlp()


    demo.launch(
        server_name = "0.0.0.0",
        server_port = port,
        share       = False,
    )