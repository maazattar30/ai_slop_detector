"""
MiQ AI Content Detector — V1
Hugging Face Space deployment
Author: Maaz Attar, MiQ Digital
"""

import os
import json
import math
import io
import base64
import subprocess
import tempfile
import warnings
import hashlib
import time
warnings.filterwarnings('ignore')

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

# ── Admin password (hash-based, set via HF Secret) ─────────────────────────
# In HF Space: Settings → Variables and secrets → Add secret: ADMIN_PASSWORD
ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', '')
GROQ_API_KEY        = os.environ.get('GROQ_API_KEY', '')

def check_password(password: str) -> bool:
    if not ADMIN_PASSWORD_HASH:
        return True  # No password set = open (set one in HF secrets!)
    h = hashlib.sha256(password.encode()).hexdigest()
    return h == ADMIN_PASSWORD_HASH

# ── Lazy imports (loaded once on first run) ──────────────────────────────────
_siglip_processor = None
_siglip_model     = None
_device           = None

def get_siglip():
    global _siglip_processor, _siglip_model, _device
    if _siglip_model is None:
        import torch
        from transformers import AutoProcessor, AutoModel
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _siglip_processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
        _siglip_model     = AutoModel.from_pretrained('google/siglip-base-patch16-224').to(_device)
        _siglip_model.eval()
    return _siglip_processor, _siglip_model, _device

# ══════════════════════════════════════════════════════════════════════════════
#  MODULE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def download_video(url, out_path):
    for client in ['web', 'android', 'ios']:
        cmd = [
            'yt-dlp',
            '--extractor-args', f'youtube:player_client={client}',
            '-f', 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',
            '-o', out_path, '--no-playlist', '--quiet', url
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            return True
    return False


def extract_frames(video_path, work_dir, n_llm=5, resolution=1024):
    frames_dir = os.path.join(work_dir, 'frames')
    all_dir    = os.path.join(work_dir, 'all_frames')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)

    subprocess.run([
        'ffmpeg', '-i', video_path,
        '-vf', f'fps=0.5,scale={resolution}:-1',
        '-q:v', '2',
        os.path.join(all_dir, 'frame_%04d.jpg'),
        '-y', '-loglevel', 'quiet'
    ], check=True)

    all_files = sorted([
        os.path.join(all_dir, f) for f in os.listdir(all_dir) if f.endswith('.jpg')
    ])

    indices   = np.linspace(0, len(all_files)-1, n_llm, dtype=int)
    llm_files = []
    for i, idx in enumerate(indices):
        dst = os.path.join(frames_dir, f'llm_{i:02d}.jpg')
        img = Image.open(all_files[idx]).convert('RGB')
        w, h = img.size
        img  = img.resize((768, int(h*768/w)), Image.LANCZOS)
        img.save(dst, 'JPEG', quality=85)
        llm_files.append(dst)

    return all_files, llm_files


def compute_audio_features(video_path, work_dir):
    import librosa
    audio_path = os.path.join(work_dir, 'audio.wav')
    subprocess.run([
        'ffmpeg', '-i', video_path, '-vn',
        '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
        audio_path, '-y', '-loglevel', 'quiet'
    ], check=True)

    y, sr = librosa.load(audio_path, sr=22050)
    feats = {}

    # Pitch
    f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=600,
                                       frame_length=2048, hop_length=512)
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
    feats['audio_pitch_std_hz']   = float(np.std(voiced_f0))   if len(voiced_f0) > 10 else 0.0
    feats['audio_pitch_range_hz'] = float(np.ptp(voiced_f0))   if len(voiced_f0) > 10 else 0.0
    feats['audio_voiced_ratio']   = float(np.mean(voiced_flag))

    # Silence
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    feats['audio_silence_ratio'] = float(np.mean(rms < 0.01))

    # Harmonic
    y_harm, y_perc = librosa.effects.hpss(y)
    he = float(np.mean(y_harm**2))
    pe = float(np.mean(y_perc**2))
    feats['audio_harmonic_ratio']    = float(he / (he + pe + 1e-10) * 10)
    feats['audio_percussive_energy'] = pe

    # Spectral
    spec    = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs   = librosa.fft_frequencies(sr=sr)
    bw      = librosa.feature.spectral_bandwidth(S=spec, sr=sr, freq=freqs)
    rolloff = librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.85)
    flat    = librosa.feature.spectral_flatness(S=spec)
    feats['audio_spectral_bandwidth_mean'] = float(np.mean(bw))
    feats['audio_spectral_rolloff_mean']   = float(np.mean(rolloff))
    feats['audio_spectral_flatness_std']   = float(np.std(flat))

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    feats['audio_tempo_bpm'] = float(tempo[0] if hasattr(tempo, '__len__') else tempo)
    if len(beats) > 2:
        ibi = np.diff(beats)
        feats['audio_tempo_regularity'] = float(1 - np.std(ibi)/(np.mean(ibi)+1e-8))
    else:
        feats['audio_tempo_regularity'] = 0.0

    feats['speech_detected'] = bool(
        feats['audio_voiced_ratio'] > 0.1 and feats['audio_pitch_std_hz'] > 5
    )
    return feats


def compute_texture_compression(all_frames):
    import cv2
    chrom, sat_cv, comp = [], [], []
    for fpath in all_frames[::3]:
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue
        b, g, r = cv2.split(img_bgr)
        er = cv2.Canny(r, 50, 150).astype(float)
        eg = cv2.Canny(g, 50, 150).astype(float)
        eb = cv2.Canny(b, 50, 150).astype(float)
        chrom.append((np.mean(np.abs(er-eg)) + np.mean(np.abs(er-eb)))/2)

        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat  = hsv[:,:,1].astype(float)
        sat_cv.append(np.std(sat) / (np.mean(sat)+1e-8))

        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img_pil.save(buf, 'JPEG', quality=50)
        buf.seek(0)
        comp_arr = np.array(Image.open(buf).convert('RGB'))
        orig_arr = np.array(img_pil)
        comp.append(np.mean(np.abs(orig_arr.astype(float) - comp_arr.astype(float))))

    return {
        'tex_chromatic_aberration':          float(np.mean(chrom)) if chrom else 0.0,
        'tex_saturation_cv':                 float(np.mean(sat_cv)) if sat_cv else 0.0,
        'compress_compression_artifact_mean':float(np.mean(comp))  if comp  else 0.0,
        'compress_compression_artifact_std': float(np.std(comp))   if comp  else 0.0,
    }


def compute_motion(all_frames):
    import cv2
    deltas, prev = [], None
    for fpath in all_frames:
        img = cv2.imread(fpath)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        if prev is not None:
            deltas.append(np.mean(np.abs(gray - prev)))
        prev = gray
    if len(deltas) > 2:
        arr = np.array(deltas)
        cv  = float(np.std(arr)/(np.mean(arr)+1e-8))
        return {
            'motion_frame_delta_mean': float(np.mean(arr)),
            'motion_frame_delta_std':  float(np.std(arr)),
            'motion_frame_delta_cv':   cv,
            'motion_cut_regularity':   float(1 - cv),
        }
    return {k: 0.0 for k in ['motion_frame_delta_mean','motion_frame_delta_std',
                               'motion_frame_delta_cv','motion_cut_regularity']}


def compute_optical_flow(all_frames):
    import cv2
    mags, angles, prev = [], [], None
    for fpath in all_frames[::2]:
        img = cv2.imread(fpath)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mags.append(float(np.mean(mag)))
            angles.append(ang[::8,::8].flatten())
        prev = gray
    if len(mags) > 2:
        all_ang = np.concatenate(angles)
        hist, _ = np.histogram(all_ang, bins=18, range=(0, 2*np.pi), density=True)
        hist   += 1e-10
        entropy = float(-np.sum(hist * np.log(hist+1e-10)))
        return {
            'raft_raft_temporal_entropy':     float(entropy * 0.35),  # rescale to match RAFT range
            'raft_raft_mean_magnitude':       float(np.mean(mags)),
            'raft_raft_direction_consistency':float(np.max(hist)/np.sum(hist)*18),
            'raft_raft_motion_smoothness':    float(1 - np.std(mags)/(np.mean(mags)+1e-8)),
        }
    return {k: 0.0 for k in ['raft_raft_temporal_entropy','raft_raft_mean_magnitude',
                               'raft_raft_direction_consistency','raft_raft_motion_smoothness']}


def compute_siglip(all_frames):
    import torch
    processor, model, device = get_siglip()
    AI_PROMPTS   = ['photorealistic AI generated image', 'AI generated digital art',
                    'synthetic computer generated video frame']
    REAL_PROMPTS = ['real photograph', 'authentic video frame from a camera',
                    'genuine human filmed video']
    all_prompts  = AI_PROMPTS + REAL_PROMPTS
    ai_probs = []
    with torch.no_grad():
        for fpath in all_frames[::4]:
            img     = Image.open(fpath).convert('RGB')
            inputs  = processor(text=all_prompts, images=[img]*len(all_prompts),
                                return_tensors='pt', padding='max_length').to(device)
            outputs = model(**inputs)
            logits  = outputs.logits_per_image[0]
            ai_l    = logits[:len(AI_PROMPTS)].mean()
            re_l    = logits[len(AI_PROMPTS):].mean()
            prob    = torch.softmax(torch.stack([ai_l, re_l]), dim=0)[0].item()
            ai_probs.append(prob)
    if ai_probs:
        return {
            'siglip_siglip_ai_mean':       float(np.mean(ai_probs)),
            'siglip_siglip_ai_max':        float(np.max(ai_probs)),
            'siglip_siglip_high_prob_frac':float(np.mean(np.array(ai_probs) > 0.5)),
        }
    return {'siglip_siglip_ai_mean':0.5,'siglip_siglip_ai_max':0.5,'siglip_siglip_high_prob_frac':0.5}


def compute_text_features(title, desc):
    AI_KEYWORDS  = ['ai','artificial intelligence','generated','sora','runway','gen-3','gen3',
                    'midjourney','kling','pika','luma','haiper','synthesia','deepfake',
                    'text to video','text-to-video','stable diffusion','dall-e','chatgpt']
    SLOP_KEYWORDS= ['viral','mind-blowing','shocking','unbelievable','must watch',
                    'insane','crazy','epic','incredible','you won\'t believe','wait for it']
    txt = (title + ' ' + desc).lower()
    return {
        'text_text_ai_keyword_count':    float(sum(1 for k in AI_KEYWORDS  if k in txt)),
        'text_text_ai_probability':      float(min(sum(1 for k in AI_KEYWORDS if k in txt)/3, 1.0)),
        'text_title_slop_keyword_count': float(sum(1 for k in SLOP_KEYWORDS if k in txt)),
        'text_title_exclamation':        float('!' in title),
        'text_text_lexical_diversity':   float(len(set(txt.split()))/(len(txt.split())+1e-8)) if txt.strip() else 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EVIDENCE BRIEF + SCORING
# ══════════════════════════════════════════════════════════════════════════════

CALIBRATION = {
    'siglip_siglip_ai_mean':              (0.640, 0.889, True,  1.52),
    'raft_raft_temporal_entropy':         (1.515, 1.964, True,  0.89),
    'tex_chromatic_aberration':           (7.848, 5.812, False, 0.81),
    'tex_saturation_cv':                  (0.850, 0.586, False, 0.82),
    'audio_harmonic_ratio':               (2.041, 1.004, False, 0.99),
    'compress_compression_artifact_mean': (1.411, 1.021, False, 0.84),
    'audio_spectral_bandwidth_mean':      (1681,  1262,  False, 0.80),
    'audio_silence_ratio':                (0.102, 0.301, True,  0.69),
    'audio_tempo_regularity':             (0.951, 0.754, False, 0.70),
    'motion_frame_delta_cv':              (0.577, 0.738, True,  0.54),
    'audio_pitch_std_hz':                 (142.6, 69.5,  False, 0.73),
    'audio_pitch_range_hz':               (911.3, 341.6, False, 1.16),
}

FEATURE_META = {
    'siglip_siglip_ai_mean':              ('SigLIP AI Probability',        'Real: 0.45–0.83 | AI: 0.62–1.00',   'Vision-language model zero-shot AI detection'),
    'raft_raft_temporal_entropy':         ('Optical Flow Entropy',          'Real: 0.34–0.77 | AI: 0.32–0.92',   'Chaos in frame-to-frame motion directions'),
    'tex_chromatic_aberration':           ('Chromatic Aberration',          'Real: 3.30–12.29 | AI: 2.93–8.64',  'Lens R/G/B channel misalignment (real cameras have this)'),
    'tex_saturation_cv':                  ('Saturation Variation',          'Real: 0.28–1.61 | AI: 0.28–0.92',   'Colour variation — real scenes are uneven, AI is uniform'),
    'audio_harmonic_ratio':               ('Audio Harmonic Ratio',          'Real: 0.80–4.66 | AI: 0.00–2.92',   'Human speech has strong harmonic overtones'),
    'compress_compression_artifact_mean': ('Compression Residual',          'Real: 0.66–2.60 | AI: 0.72–1.87',   'Real frames resist re-compression more than AI frames'),
    'audio_spectral_bandwidth_mean':      ('Audio Spectral Bandwidth',      'Real: 1337–4958 Hz | AI: 0–3820 Hz','Frequency width of audio — real is broader'),
    'audio_silence_ratio':                ('Audio Silence Ratio',           'Real: 0.00–0.24 | AI: 0.00–1.00',   'AI videos (Sora, Midjourney) often have no audio'),
    'audio_tempo_regularity':             ('Tempo Regularity',              'Real: 0.93–0.97 | AI: 0.00–0.95',   'Real audio has natural timing variation'),
    'motion_frame_delta_cv':              ('Motion Irregularity',           'Real: 0.23–1.06 | AI: 0.46–1.56',   'AI motion is either static or uniformly erratic'),
    'audio_pitch_std_hz':                 ('Pitch Variation (Std)',         'Real: 24–429 Hz | AI: 0–266 Hz',    'Natural speech prosody — pitch varies continuously'),
    'audio_pitch_range_hz':               ('Pitch Range',                   'Real: 244–1899 Hz | AI: 0–840 Hz',  'Human speakers cover a wide natural pitch range'),
}

def label_signal(feat, value, rm, am, hia, d, speech):
    if feat in ('audio_pitch_std_hz','audio_pitch_range_hz') and not speech:
        return 'N/A', 0.5
    lo, hi = min(rm, am), max(rm, am)
    span   = hi - lo
    if span < 1e-10:
        return 'NEUTRAL', 0.5
    clipped  = np.clip(value, lo - 0.5*span, hi + 0.5*span)
    raw      = (clipped - lo) / (hi - lo)
    ai_prob  = float(np.clip(raw if hia else 1-raw, 0, 1))
    if feat in ('audio_pitch_std_hz','audio_pitch_range_hz') and speech and ai_prob < 0.7:
        return 'NEUTRAL', 0.5
    if   ai_prob >= 0.75: label = 'STRONG AI'
    elif ai_prob >= 0.55: label = 'MODERATE AI'
    elif ai_prob >= 0.45: label = 'NEUTRAL'
    elif ai_prob >= 0.25: label = 'MODERATE REAL'
    else:                 label = 'STRONG REAL'
    return label, ai_prob

def build_evidence(all_feats, speech):
    lines, probs, weights, signal_data = [], [], [], []
    for feat, (rm, am, hia, d) in CALIBRATION.items():
        val = all_feats.get(feat)
        if val is None:
            continue
        label, ai_prob = label_signal(feat, val, rm, am, hia, d, speech)
        short, ranges, defn = FEATURE_META.get(feat, (feat,'',''))
        lines.append(f'  {feat} = {val:.4f}  →  [{label}]  |  {ranges}  |  {defn}')
        signal_data.append({'feature': feat, 'short': short, 'value': val,
                            'label': label, 'ai_prob': ai_prob, 'ranges': ranges, 'defn': defn})
        if 'N/A' not in label:
            probs.append(ai_prob)
            weights.append(d)
    w   = np.array(weights)
    w  /= w.sum()
    mod = float(np.dot(probs, w) * 100) if probs else 50.0
    return '\n'.join(lines), mod, signal_data


# ══════════════════════════════════════════════════════════════════════════════
#  LLM JUDGE
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert AI video forensics analyst for MiQ Digital, a programmatic advertising company.

Classify YouTube videos into one of four buckets:
- Bucket 1 (Human Only): Entirely human-created. Real camera, real people, natural audio.
- Bucket 2 (Augmented Human): Primarily human. AI used only as a tool (editing, grading). Human creative control.
- Bucket 3 (Augmented AI): AI-generated output with human direction/prompting/curation.
- Bucket 4 (Fully Autogenerated): Entirely AI-generated. Text-to-video, AI avatars, TTS, deepfakes, content-farm slop.

You receive: signal evidence brief, module score, video frames, title, description.
Use signals as primary evidence. Use frames to confirm or override.
Watch for: real human faces with natural blinking/eye movement, motion physics, lighting consistency, background stability.

Caveats:
- High-production real content (clean offices, studios) can score high on AI signals — look for real faces
- AI video with background music can have real-sounding audio — weight visual evidence more
- CGI in real films is NOT AI-generated for our purposes
- Deepfakes: real body motion but AI face — look for unnatural skin texture, face boundary artifacts

Respond ONLY with valid JSON, no text before or after:
{
  "bucket": <1|2|3|4>,
  "ai_score": <0-100>,
  "confidence": <0.0-1.0>,
  "bucket_name": "<Human Only|Augmented Human|Augmented AI|Fully Autogenerated>",
  "primary_signals": ["<signal>", "<signal>", "<signal>"],
  "visual_evidence": ["<observation>", "<observation>"],
  "module_override": <true|false>,
  "override_reason": "<reason or empty string>",
  "explanation": "<2-4 sentence plain English for a client>",
  "red_flags": ["<AI artifact>"]
}"""

def call_llm(title, desc, module_score, evidence_brief, encoded_frames, frame_labels):
    from groq import Groq
    client   = Groq(api_key=GROQ_API_KEY)
    user_txt = f"""VIDEO ANALYSIS REQUEST
Title: {title}
Description: {desc or '(none)'}
Module Score: {module_score:.1f}/100
(0–30=strongly real, 30–50=likely real, 50–70=uncertain, 70–85=likely AI, 85–100=strongly AI)

SIGNAL EVIDENCE:
{evidence_brief}

Examine the {len(encoded_frames)} image(s) attached and classify."""

    content = [{"type":"text","text":user_txt}]
    for i, (b64, label) in enumerate(zip(encoded_frames, frame_labels)):
        content.append({"type":"text","text":f"Image {i+1}: {label}"})
        content.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})

    resp = client.chat.completions.create(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":content}],
        temperature=0.1, max_tokens=1200
    )
    raw = resp.choices[0].message.content.strip()
    if '```' in raw:
        raw = raw.split('```')[1]
        if raw.startswith('json'): raw = raw[4:]
    return json.loads(raw.strip())


def prepare_frames_for_llm(llm_files, max_images=5):
    """Distribute N frames into max_images slots, stitching overflow into last slot."""
    def enc(img):
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=82)
        return base64.b64encode(buf.getvalue()).decode()

    def resize(path, w=768):
        img = Image.open(path).convert('RGB')
        ow, oh = img.size
        return img.resize((w, int(oh*w/ow)), Image.LANCZOS)

    def stitch(paths):
        imgs = [Image.open(p).convert('RGB').resize((512,288),Image.LANCZOS) for p in paths]
        cols = math.ceil(math.sqrt(len(imgs)))
        rows = math.ceil(len(imgs)/cols)
        grid = Image.new('RGB',(512*cols,288*rows),(20,20,20))
        draw = ImageDraw.Draw(grid)
        for i,img in enumerate(imgs):
            r,c = divmod(i,cols)
            grid.paste(img,(c*512,r*288))
            draw.rectangle([c*512+2,r*288+2,c*512+48,r*288+18],fill=(0,0,0))
            draw.text((c*512+4,r*288+3),f'F{i+1}',fill=(255,255,255))
        return grid

    n = len(llm_files)
    encoded, labels = [], []

    if n <= max_images:
        for i,fp in enumerate(llm_files):
            pct = int(i*100/max(n-1,1))
            encoded.append(enc(resize(fp)))
            labels.append(f'Frame {i+1} at {pct}% of video')
    else:
        idxs = [int(i*(n-1)/(max_images-1)) for i in range(max_images-1)]
        used = set(idxs)
        for slot,idx in enumerate(idxs):
            pct = int(idx*100/max(n-1,1))
            encoded.append(enc(resize(llm_files[idx])))
            labels.append(f'Frame {idx+1} at {pct}% of video')
        remaining = [llm_files[i] for i in range(n) if i not in used]
        pcts = [int(i*100/max(n-1,1)) for i in range(n) if i not in used]
        encoded.append(enc(stitch(remaining)))
        labels.append(f'Grid: {len(remaining)} frames at {pcts[0]}%–{pcts[-1]}%')

    return encoded, labels


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(password, url, title, desc, progress=gr.Progress()):
    # Auth
    if not check_password(password):
        return None, None, "❌ Incorrect admin password.", None

    if not url or not url.strip():
        return None, None, "❌ Please enter a video URL.", None

    if not GROQ_API_KEY:
        return None, None, "❌ GROQ_API_KEY not set in HF Space secrets.", None

    title = title.strip() if title else url
    desc  = desc.strip()  if desc  else ''

    with tempfile.TemporaryDirectory() as work_dir:
        try:
            # 1 — Download
            progress(0.05, "📥 Downloading video...")
            video_path = os.path.join(work_dir, 'video.mp4')
            if not download_video(url, video_path):
                return None, None, "❌ Failed to download video. Check URL.", None

            # 2 — Frames
            progress(0.15, "🎞 Extracting frames...")
            all_frames, llm_frames = extract_frames(video_path, work_dir)

            # 3 — Audio
            progress(0.25, "🔊 Computing audio features...")
            audio_feats = compute_audio_features(video_path, work_dir)

            # 4 — Texture
            progress(0.40, "🎨 Computing texture & compression...")
            tex_feats = compute_texture_compression(all_frames)

            # 5 — Motion
            progress(0.50, "🏃 Computing motion features...")
            mot_feats = compute_motion(all_frames)

            # 6 — Optical flow
            progress(0.60, "🌊 Computing optical flow...")
            flow_feats = compute_optical_flow(all_frames)

            # 7 — SigLIP
            progress(0.72, "🤖 Running SigLIP classifier...")
            sig_feats = compute_siglip(all_frames)

            # 8 — Text
            progress(0.80, "📝 Analysing text signals...")
            txt_feats = compute_text_features(title, desc)

            # 9 — Evidence brief
            progress(0.85, "📊 Building evidence brief...")
            all_feats = {**audio_feats, **tex_feats, **mot_feats,
                         **flow_feats, **sig_feats, **txt_feats}
            speech = audio_feats.get('speech_detected', False)
            evidence_brief, module_score, signal_data = build_evidence(all_feats, speech)

            # 10 — Encode frames
            progress(0.88, "🖼 Preparing frames for LLM...")
            encoded_frames, frame_labels = prepare_frames_for_llm(llm_frames, max_images=5)

            # 11 — LLM
            progress(0.92, "🧠 Calling Llama 4 Scout (Groq)...")
            llm_result = call_llm(title, desc, module_score, evidence_brief,
                                   encoded_frames, frame_labels)

            # 12 — Blend
            progress(0.97, "✅ Computing final verdict...")
            llm_score   = llm_result.get('ai_score', module_score)
            final_score = round(0.55 * module_score + 0.45 * llm_score, 1)
            final_bucket= llm_result.get('bucket', 4 if final_score > 60 else 1)

            # Collect frame images for display
            frame_imgs = [Image.open(f) for f in llm_frames]

            progress(1.0, "Done!")
            return (
                build_result_html(title, url, module_score, llm_score, final_score,
                                  final_bucket, llm_result, signal_data, encoded_frames, frame_labels),
                frame_imgs,
                f"✅ Analysis complete — Bucket {final_bucket}",
                signal_data
            )

        except Exception as e:
            import traceback
            return None, None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", None


# ══════════════════════════════════════════════════════════════════════════════
#  HTML DASHBOARD BUILDER
# ══════════════════════════════════════════════════════════════════════════════

BUCKET_COLOR  = {1:'#00C896', 2:'#3B82F6', 3:'#F59E0B', 4:'#EF4444'}
BUCKET_EMOJI  = {1:'🟢', 2:'🔵', 3:'🟠', 4:'🔴'}
BUCKET_NAME   = {1:'Human Only', 2:'Augmented Human', 3:'Augmented AI', 4:'Fully Autogenerated'}
SIGNAL_COLOR  = {
    'STRONG AI':    '#EF4444',
    'MODERATE AI':  '#F59E0B',
    'NEUTRAL':      '#6B7280',
    'MODERATE REAL':'#3B82F6',
    'STRONG REAL':  '#00C896',
    'N/A':          '#9CA3AF',
}

def score_bar(score, color, max_val=100):
    pct = min(max(score, 0), max_val) / max_val * 100
    return f"""<div style="background:#1E293B;border-radius:6px;height:10px;width:100%;margin:6px 0">
      <div style="background:{color};width:{pct}%;height:10px;border-radius:6px;
                  transition:width 0.8s ease"></div></div>"""

def build_result_html(title, url, module_score, llm_score, final_score,
                      bucket, llm_result, signal_data, encoded_frames, frame_labels):
    bc   = BUCKET_COLOR[bucket]
    bn   = BUCKET_NAME[bucket]
    be   = BUCKET_EMOJI[bucket]
    conf = llm_result.get('confidence', 0.7)

    # Signal rows
    sig_rows = ''
    for s in signal_data:
        lc  = SIGNAL_COLOR.get(s['label'], '#6B7280')
        bar = score_bar(s['ai_prob']*100, lc)
        sig_rows += f"""
        <tr>
          <td style="padding:10px 14px;color:#CBD5E1;font-size:13px;font-weight:500">{s['short']}</td>
          <td style="padding:10px 14px;font-family:'JetBrains Mono',monospace;color:#F1F5F9;font-size:13px">{s['value']:.4f}</td>
          <td style="padding:10px 14px">
            <span style="background:{lc}22;color:{lc};border:1px solid {lc}44;
                         padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;
                         letter-spacing:0.05em">{s['label']}</span>
          </td>
          <td style="padding:10px 14px">
            <div style="min-width:120px">{bar}</div>
          </td>
          <td style="padding:10px 14px;color:#64748B;font-size:12px">{s['ranges']}</td>
          <td style="padding:10px 14px;color:#64748B;font-size:12px">{s['defn']}</td>
        </tr>"""

    # Evidence items
    def list_items(items, color='#CBD5E1'):
        return ''.join(f'<li style="margin:5px 0;color:{color}">{i}</li>' for i in items) if items else '<li style="color:#475569">None</li>'

    prim_html  = list_items(llm_result.get('primary_signals', []))
    vis_html   = list_items(llm_result.get('visual_evidence', []))
    flags_html = list_items(llm_result.get('red_flags', []), '#F87171')

    override_html = ''
    if llm_result.get('module_override'):
        override_html = f"""
        <div style="background:#F59E0B11;border:1px solid #F59E0B44;border-left:3px solid #F59E0B;
                    padding:12px 16px;border-radius:6px;margin:16px 0">
          <span style="color:#F59E0B;font-weight:700">⚠ LLM override: </span>
          <span style="color:#CBD5E1">{llm_result.get('override_reason','')}</span>
        </div>"""

    # Frames row
    frames_html = ''
    for i, (b64, label) in enumerate(zip(encoded_frames, frame_labels)):
        frames_html += f"""
        <div style="flex:1;min-width:0">
          <img src="data:image/jpeg;base64,{b64}"
               style="width:100%;border-radius:6px;border:1px solid #1E293B;display:block"/>
          <div style="color:#64748B;font-size:11px;margin-top:4px;text-align:center">{label}</div>
        </div>"""

    # Bucket meter (4 segments)
    bucket_meter = ''
    for b in [1,2,3,4]:
        active = 'opacity:1' if b == bucket else 'opacity:0.25'
        bucket_meter += f"""
        <div style="flex:1;background:{BUCKET_COLOR[b]};padding:8px 4px;
                    border-radius:4px;text-align:center;{active}">
          <div style="font-size:18px">{BUCKET_EMOJI[b]}</div>
          <div style="font-size:10px;color:white;font-weight:700;margin-top:2px">
            B{b}
          </div>
        </div>"""

    html = f"""
<div style="font-family:'IBM Plex Sans',system-ui,sans-serif;background:#0F172A;
            color:#F1F5F9;padding:28px;border-radius:12px;max-width:1200px;margin:0 auto">

  <!-- Header -->
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
    <div>
      <div style="font-size:11px;color:#475569;letter-spacing:0.1em;text-transform:uppercase;
                  margin-bottom:4px">MiQ AI Content Detector · V1</div>
      <h2 style="margin:0;font-size:20px;color:#F1F5F9">{title}</h2>
      <a href="{url}" style="color:#3B82F6;font-size:12px;text-decoration:none">{url[:70]}...</a>
    </div>
    <div style="text-align:right">
      <div style="font-size:11px;color:#475569;margin-bottom:4px">CONFIDENCE</div>
      <div style="font-size:28px;font-weight:800;color:{bc}">{conf:.0%}</div>
    </div>
  </div>

  <!-- Verdict -->
  <div style="background:{bc}0F;border:1px solid {bc}44;border-radius:10px;
              padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:20px">
    <div style="font-size:52px;line-height:1">{be}</div>
    <div style="flex:1">
      <div style="font-size:13px;color:{bc};font-weight:700;letter-spacing:0.08em;
                  text-transform:uppercase;margin-bottom:2px">VERDICT</div>
      <div style="font-size:28px;font-weight:800;color:#F1F5F9">
        Bucket {bucket} — {bn}
      </div>
    </div>
    <div style="display:flex;gap:6px;align-self:stretch">{bucket_meter}</div>
  </div>

  <!-- Score cards -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px">
    <div style="background:#1E293B;border-radius:8px;padding:16px">
      <div style="font-size:11px;color:#475569;letter-spacing:0.08em;text-transform:uppercase">
        MODULE SCORE</div>
      <div style="font-size:36px;font-weight:800;color:#3B82F6;margin:4px 0">{module_score:.1f}</div>
      {score_bar(module_score, '#3B82F6')}
      <div style="font-size:11px;color:#475569">12 computed signals · Cohen d-weighted</div>
    </div>
    <div style="background:#1E293B;border-radius:8px;padding:16px">
      <div style="font-size:11px;color:#475569;letter-spacing:0.08em;text-transform:uppercase">
        LLM SCORE</div>
      <div style="font-size:36px;font-weight:800;color:#8B5CF6;margin:4px 0">{llm_score}</div>
      {score_bar(llm_score, '#8B5CF6')}
      <div style="font-size:11px;color:#475569">Llama 4 Scout visual + evidence reasoning</div>
    </div>
    <div style="background:{bc}0F;border:1px solid {bc}33;border-radius:8px;padding:16px">
      <div style="font-size:11px;color:{bc};letter-spacing:0.08em;text-transform:uppercase">
        FINAL SCORE</div>
      <div style="font-size:36px;font-weight:800;color:{bc};margin:4px 0">{final_score}</div>
      {score_bar(final_score, bc)}
      <div style="font-size:11px;color:#475569">55% module · 45% LLM blend</div>
    </div>
  </div>

  {override_html}

  <!-- Explanation -->
  <div style="background:#1E293B;border-left:3px solid #3B82F6;border-radius:6px;
              padding:16px 20px;margin-bottom:20px">
    <div style="font-size:11px;color:#3B82F6;font-weight:700;letter-spacing:0.08em;
                text-transform:uppercase;margin-bottom:8px">CLIENT EXPLANATION</div>
    <p style="margin:0;color:#CBD5E1;line-height:1.7">{llm_result.get('explanation','')}</p>
  </div>

  <!-- Evidence grid -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px">
    <div style="background:#1E293B;border-radius:8px;padding:16px">
      <div style="font-size:11px;color:#3B82F6;font-weight:700;letter-spacing:0.08em;
                  text-transform:uppercase;margin-bottom:10px">PRIMARY SIGNALS</div>
      <ul style="margin:0;padding-left:18px">{prim_html}</ul>
    </div>
    <div style="background:#1E293B;border-radius:8px;padding:16px">
      <div style="font-size:11px;color:#00C896;font-weight:700;letter-spacing:0.08em;
                  text-transform:uppercase;margin-bottom:10px">VISUAL EVIDENCE</div>
      <ul style="margin:0;padding-left:18px">{vis_html}</ul>
    </div>
    <div style="background:#1E293B;border-radius:8px;padding:16px">
      <div style="font-size:11px;color:#EF4444;font-weight:700;letter-spacing:0.08em;
                  text-transform:uppercase;margin-bottom:10px">AI RED FLAGS</div>
      <ul style="margin:0;padding-left:18px">{flags_html}</ul>
    </div>
  </div>

  <!-- Signal table -->
  <div style="background:#1E293B;border-radius:8px;margin-bottom:20px;overflow:hidden">
    <div style="padding:14px 20px;border-bottom:1px solid #0F172A">
      <span style="font-size:13px;font-weight:700;color:#F1F5F9;
                   letter-spacing:0.05em;text-transform:uppercase">Signal Breakdown</span>
      <span style="margin-left:10px;font-size:11px;color:#475569">
        12 features · calibrated on 20-video dataset</span>
    </div>
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="background:#0F172A">
          <th style="padding:10px 14px;text-align:left;color:#475569;font-size:11px;
                     font-weight:700;letter-spacing:0.08em;text-transform:uppercase">Feature</th>
          <th style="padding:10px 14px;text-align:left;color:#475569;font-size:11px;
                     font-weight:700;letter-spacing:0.08em;text-transform:uppercase">Value</th>
          <th style="padding:10px 14px;text-align:left;color:#475569;font-size:11px;
                     font-weight:700;letter-spacing:0.08em;text-transform:uppercase">Assessment</th>
          <th style="padding:10px 14px;text-align:left;color:#475569;font-size:11px;
                     font-weight:700;letter-spacing:0.08em;text-transform:uppercase">AI Probability</th>
          <th style="padding:10px 14px;text-align:left;color:#475569;font-size:11px;
                     font-weight:700;letter-spacing:0.08em;text-transform:uppercase">Observed Ranges</th>
          <th style="padding:10px 14px;text-align:left;color:#475569;font-size:11px;
                     font-weight:700;letter-spacing:0.08em;text-transform:uppercase">What it measures</th>
        </tr>
      </thead>
      <tbody>{sig_rows}</tbody>
    </table>
    </div>
  </div>

  <!-- Frames -->
  <div style="background:#1E293B;border-radius:8px;padding:16px">
    <div style="font-size:11px;color:#F1F5F9;font-weight:700;letter-spacing:0.08em;
                text-transform:uppercase;margin-bottom:12px">
      Sampled Frames Sent to LLM</div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">{frames_html}</div>
  </div>

  <div style="margin-top:16px;text-align:center;color:#334155;font-size:11px">
    MiQ AI Content Detector V1 &nbsp;·&nbsp; 12 signals &nbsp;·&nbsp;
    Llama 4 Scout (Groq) &nbsp;·&nbsp; maazattar.com
  </div>
</div>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

body, .gradio-container {
    background: #0A0F1E !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

/* Header */
.app-header {
    text-align: center;
    padding: 48px 0 32px;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 32px;
}
.app-header h1 {
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    color: #3B82F6 !important;
    text-transform: uppercase !important;
    margin: 0 0 12px !important;
}
.app-header h2 {
    font-size: 36px !important;
    font-weight: 300 !important;
    color: #F1F5F9 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.02em !important;
}
.app-header p {
    color: #475569 !important;
    font-size: 14px !important;
    margin: 0 !important;
}

/* Form card */
.form-card {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}

/* Inputs */
.gradio-container input[type=text], .gradio-container input[type=password],
.gradio-container textarea {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: #3B82F6 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px #3B82F620 !important;
}
label span {
    color: #94A3B8 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* Run button */
.run-btn {
    background: linear-gradient(135deg, #1D4ED8, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.run-btn:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* Status */
.status-box {
    background: #0F172A !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    color: #64748B !important;
    font-size: 13px !important;
}

/* Tabs */
.tab-nav button {
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
    color: #3B82F6 !important;
    border-bottom-color: #3B82F6 !important;
}

/* Hide default Gradio chrome */
footer { display: none !important; }
.svelte-1ed2p3z { border: none !important; }
"""

def build_ui():
    with gr.Blocks(css=CSS, title="MiQ AI Detector") as demo:

        # Header
        gr.HTML("""
        <div class="app-header">
          <h1>MiQ Digital · Product Analytics</h1>
          <h2>AI Content Detector</h2>
          <p>Classify YouTube inventory into Human / Augmented / AI-Generated buckets</p>
        </div>
        """)

        with gr.Row():
            # Left — inputs
            with gr.Column(scale=1):
                gr.HTML('<div class="form-card">')
                gr.HTML('<div style="font-size:11px;color:#3B82F6;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px">Analysis Input</div>')

                password = gr.Textbox(
                    label="Admin Password",
                    type="password",
                    placeholder="Enter access password...",
                )
                url = gr.Textbox(
                    label="YouTube URL  *",
                    placeholder="https://www.youtube.com/watch?v=...",
                )
                title = gr.Textbox(
                    label="Video Title  (optional)",
                    placeholder="e.g. Life at MiQ",
                )
                desc = gr.Textbox(
                    label="Description  (optional)",
                    placeholder="Paste description for stronger text signal analysis...",
                    lines=3,
                )
                run_btn = gr.Button("▶  Run Analysis", elem_classes="run-btn")
                status  = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes="status-box",
                    value="Waiting for input..."
                )
                gr.HTML('</div>')

                # Info card
                gr.HTML("""
                <div style="background:#1E293B;border-radius:10px;padding:20px;margin-top:4px">
                  <div style="font-size:11px;color:#3B82F6;font-weight:700;letter-spacing:0.1em;
                              text-transform:uppercase;margin-bottom:14px">Bucket Reference</div>
                  <div style="display:flex;flex-direction:column;gap:8px">
                    <div style="display:flex;align-items:center;gap:10px">
                      <span style="font-size:18px">🟢</span>
                      <div><b style="color:#00C896">B1 Human Only</b>
                        <div style="color:#64748B;font-size:11px">Real camera, natural audio, no AI</div></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                      <span style="font-size:18px">🔵</span>
                      <div><b style="color:#3B82F6">B2 Augmented Human</b>
                        <div style="color:#64748B;font-size:11px">Human-made, AI tools used for editing</div></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                      <span style="font-size:18px">🟠</span>
                      <div><b style="color:#F59E0B">B3 Augmented AI</b>
                        <div style="color:#64748B;font-size:11px">AI output with human direction</div></div>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px">
                      <span style="font-size:18px">🔴</span>
                      <div><b style="color:#EF4444">B4 Fully Autogenerated</b>
                        <div style="color:#64748B;font-size:11px">Text-to-video, deepfake, TTS, slop</div></div>
                    </div>
                  </div>
                </div>
                """)

            # Right — output
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Dashboard"):
                        result_html = gr.HTML(
                            value='<div style="background:#1E293B;border-radius:10px;padding:60px;'
                                  'text-align:center;color:#334155">'
                                  '<div style="font-size:32px;margin-bottom:12px">🎬</div>'
                                  '<div style="font-size:14px">Run an analysis to see the dashboard</div></div>'
                        )
                    with gr.Tab("🖼 Frames"):
                        frame_gallery = gr.Gallery(
                            label="Sampled Frames",
                            columns=3,
                            height=400,
                            show_label=False,
                        )

        # Wire up
        run_btn.click(
            fn=run_pipeline,
            inputs=[password, url, title, desc],
            outputs=[result_html, frame_gallery, status, gr.State()],
            show_progress=True,
        )

    return demo

if __name__ == '__main__':
    app = build_ui()
    app.launch(share=False)
