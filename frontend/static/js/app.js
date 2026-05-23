/* ═══════════════════════════════════════════════════════════════
   ImgAuth AI — Frontend Logic
   ─ Binary verdict: Likely AI-Generated vs Likely Authentic
   ─ Bug fix: result section scrolls into view after analysis
   ═══════════════════════════════════════════════════════════════ */

// ── Theme Toggle ──────────────────────────────────────────────────
(function () {
  const html = document.documentElement;
  const btn  = document.querySelector('[data-theme-toggle]');
  let theme  = localStorage.getItem('imgauth-theme') || 'dark';

  html.setAttribute('data-theme', theme);
  if (btn) btn.textContent = theme === 'dark' ? '🌙' : '☀️';

  if (btn) {
    btn.addEventListener('click', () => {
      theme = theme === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', theme);
      btn.textContent = theme === 'dark' ? '🌙' : '☀️';
      localStorage.setItem('imgauth-theme', theme);
    });
  }
})();

// ── Element Refs ──────────────────────────────────────────────────
const fileInput     = document.getElementById('fileInput');
const dropZone      = document.getElementById('dropZone');
const previewZone   = document.getElementById('previewZone');
const previewImg    = document.getElementById('previewImg');
const previewFilename = document.getElementById('previewFilename');
const analyzeBtn    = document.getElementById('analyzeBtn');
const clearBtn      = document.getElementById('clearBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultSection = document.getElementById('resultSection');
const newScanBtn    = document.getElementById('newScanBtn');

let selectedFile = null;

// ── File Handling ─────────────────────────────────────────────────
fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('Please select a valid image file (PNG, JPG, WEBP).', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Maximum size is 10 MB.', 'error');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src       = ev.target.result;
    previewFilename.textContent = file.name;
    dropZone.style.display   = 'none';
    previewZone.style.display = 'block';
    resultSection.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', resetToUpload);

// ── Analyze ───────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  showLoading();
  try {
    const form = new FormData();
    form.append('file', selectedFile);
    const res  = await fetch('/api/detect', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Server error');
    hideLoading();
    showResult(data);
  } catch (err) {
    hideLoading();
    showToast('Error: ' + err.message, 'error');
  }
});

newScanBtn.addEventListener('click', resetToUpload);

function resetToUpload() {
  resultSection.style.display = 'none';
  previewZone.style.display   = 'none';
  dropZone.style.display      = 'block';
  selectedFile  = null;
  fileInput.value = '';
  // Scroll back up to the upload section
  document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Loading Steps ─────────────────────────────────────────────────
let stepTimer = null;
const STEP_IDS = ['step1','step2','step3','step4','step5','step6'];

function showLoading() {
  // Reset all steps
  STEP_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = 'step';
  });
  loadingOverlay.style.display = 'flex';
  let i = 0;
  stepTimer = setInterval(() => {
    if (i < STEP_IDS.length) {
      if (i > 0) {
        const prev = document.getElementById(STEP_IDS[i - 1]);
        if (prev) prev.className = 'step done';
      }
      const cur = document.getElementById(STEP_IDS[i]);
      if (cur) cur.className = 'step active';
      i++;
    }
  }, 700);
}

function hideLoading() {
  clearInterval(stepTimer);
  loadingOverlay.style.display = 'none';
}

// ── Show Result ───────────────────────────────────────────────────
function showResult(data) {
  const aiScore  = typeof data.ai_score  === 'number' ? data.ai_score  : 50;
  const realScore = typeof data.real_score === 'number' ? data.real_score : 50;

  // ── Populate result image
  document.getElementById('resultImg').src = previewImg.src;

  // ── Binary verdict (no "Uncertain" category)
  const isAI = aiScore > 50;

  const badge   = document.getElementById('resultBadge');
  const verdict = document.getElementById('resultVerdict');

  if (isAI) {
    badge.textContent = '🔴 AI-Generated';
    badge.className   = 'verdict-badge is-ai';
    verdict.textContent = 'Likely AI-Generated';
    verdict.className   = 'verdict-heading is-ai';
  } else {
    badge.textContent = '🟢 Authentic';
    badge.className   = 'verdict-badge is-real';
    verdict.textContent = 'Likely Authentic';
    verdict.className   = 'verdict-heading is-real';
  }

  // ── Confidence label — based on distance from 50%
  const margin = Math.abs(aiScore - 50);
  let confidenceLabel;
  if      (margin >= 35) confidenceLabel = 'Very High Confidence';
  else if (margin >= 20) confidenceLabel = 'High Confidence';
  else if (margin >= 10) confidenceLabel = 'Medium Confidence';
  else                   confidenceLabel = 'Low Confidence';
  document.getElementById('resultConfidenceLevel').textContent = confidenceLabel;

  // ── Gauge bar (animate after a tick so CSS transition fires)
  setTimeout(() => {
    document.getElementById('gaugeFill').style.width = aiScore + '%';
    document.getElementById('gaugeDot').style.left   = aiScore + '%';
    document.getElementById('realPct').textContent   = realScore.toFixed(1) + '%';
    document.getElementById('aiPct').textContent     = aiScore.toFixed(1) + '%';
  }, 80);

  // ── Plain-language summary
  document.getElementById('resultSummary').textContent = buildSummary(aiScore);

  // ── "Why this result" cards
  buildWhyCards(data, aiScore);

  // ── Heatmaps
  buildHeatmaps(data);

  // ── Advanced technical section
  buildAdvanced(data);

  // ── Advanced toggle — clone to remove stale listeners
  const oldBtn = document.getElementById('advancedToggle');
  const newBtn = oldBtn.cloneNode(true);
  oldBtn.parentNode.replaceChild(newBtn, oldBtn);
  newBtn.addEventListener('click', function () {
    const body   = document.getElementById('advancedBody');
    const isOpen = body.style.display === 'block';
    body.style.display = isOpen ? 'none' : 'block';
    this.setAttribute('aria-expanded', String(!isOpen));
    this.querySelector('span').textContent = isOpen
      ? 'Show Technical Analysis'
      : 'Hide Technical Analysis';
  });

  // ── Show result section then scroll TO IT  ← BUG FIX
  resultSection.style.display = 'block';
  setTimeout(() => {
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 60);
}

// ── Build plain-language summary ──────────────────────────────────
function buildSummary(aiScore) {
  if (aiScore >= 80) return 'This image shows strong patterns typically found in AI-generated content. Multiple models flagged it with high confidence.';
  if (aiScore >= 60) return 'This image shows several patterns commonly associated with AI-generated images.';
  if (aiScore >= 50) return 'This image leans towards AI-generated, but the signal is relatively weak.';
  if (aiScore >= 35) return 'This image appears to be authentic, though some minor signals were detected.';
  return 'This image shows patterns consistent with authentic, real-world photographs.';
}

// ── Build "Why this result" cards ─────────────────────────────────
function buildWhyCards(data, aiScore) {
  // Visual Patterns
  let visualText;
  if (aiScore >= 70)
    visualText = 'Detected unusual texture patterns and edge artifacts commonly found in AI-generated images.';
  else if (aiScore >= 50)
    visualText = 'Some visual patterns are consistent with AI generation, though results are mixed.';
  else
    visualText = 'Visual patterns appear natural and consistent with authentic photographs.';
  document.getElementById('explainVisual').textContent = visualText;

  // AI Model Analysis
  let modelText;
  if (aiScore >= 70)
    modelText = 'Multiple AI detection models identified patterns strongly associated with AI-generated images.';
  else if (aiScore >= 50)
    modelText = 'AI detection models show a mild lean toward generated content.';
  else
    modelText = 'AI models detected characteristics more consistent with real, authentic photographs.';
  document.getElementById('explainModels').textContent = modelText;

  // Metadata Check
  let metaText = 'No reliable camera metadata was found. This may indicate the image was generated or heavily edited.';
  const mdLayer = (data.breakdown || []).find(b => b.layer === 'Metadata Analysis');
  if (mdLayer && mdLayer.signals) {
    const sigs = mdLayer.signals.join(' ');
    if (sigs.includes('camera EXIF') || sigs.includes('GPS'))
      metaText = 'Camera metadata was found, suggesting the image may have been captured with a real device.';
    else if (sigs.includes('Software = AI') || sigs.includes('PNG metadata key'))
      metaText = 'Metadata explicitly references AI generation tools — a strong indicator of synthetic origin.';
  }
  document.getElementById('explainMetadata').textContent = metaText;
}

// ── Build Heatmaps ────────────────────────────────────────────────
function buildHeatmaps(data) {
  const focusAreas = document.getElementById('focusAreas');
  const dfiCard    = document.getElementById('dfiCardContainer');
  const attImg     = document.getElementById('attentionHeatmapImg');
  const dfiImg     = document.getElementById('dfiHeatmapImg');

  let hasAny = false;
  const ml = data.layers && data.layers.models ? data.layers.models : {};

  if (ml.attention_heatmap) {
    attImg.src = ml.attention_heatmap;
    hasAny = true;
  }
  if (ml.dfi_heatmap) {
    dfiImg.src = ml.dfi_heatmap;
    dfiCard.style.display = 'block';
    hasAny = true;
  } else {
    dfiCard.style.display = 'none';
  }

  focusAreas.style.display = hasAny ? 'block' : 'none';
}

// ── Build Advanced Technical Section ─────────────────────────────
function buildAdvanced(data) {
  const container = document.getElementById('advancedContent');
  container.innerHTML = '';

  const fnLayer = (data.breakdown || []).find(b => b.layer === 'Filename Analysis');
  const mdLayer = (data.breakdown || []).find(b => b.layer === 'Metadata Analysis');
  const mlLayer = (data.breakdown || []).find(b => b.layer === 'AI Model and Forensic Detectors');

  const votes    = mlLayer ? (mlLayer.votes    || []) : [];
  const forensics = mlLayer ? (mlLayer.forensics || {}) : {};

  // Card 1 — Model Results
  const dlVotes = votes.filter(v => v.type === 'deep_learning');
  const modelItems = dlVotes.length
    ? dlVotes.map(v => `<li><span>${esc(v.detector)}</span><span class="adv-val">${(v.ai_prob * 100).toFixed(1)}% AI</span></li>`).join('')
    : '<li><span>No models available</span></li>';
  container.appendChild(makeAdvCard('Model Results', modelItems));

  // Card 2 — Forensic Signatures
  const fmtNum = (v, dec = 4) => (v !== undefined && v !== null) ? Number(v).toFixed(dec) : 'N/A';
  const kurtosis  = forensics.kurtosis   || {};
  const dfi       = forensics.dfi        || {};
  const fft       = forensics.fft        || {};
  const chist     = forensics.color_histogram || {};
  const jpeg      = forensics.jpeg_ghost  || {};
  const jpegNA    = jpeg.detail === 'Not a JPEG' || !jpeg.ghost_spread;
  const forensicItems = `
    <li><span>Noise Kurtosis</span>    <span class="adv-val">${fmtNum(kurtosis.kurtosis)}</span></li>
    <li><span>DFI Variance</span>      <span class="adv-val">${fmtNum(dfi.variance, 5)}</span></li>
    <li><span>FFT Spike Ratio</span>   <span class="adv-val">${fmtNum(fft.spike_ratio, 5)}</span></li>
    <li><span>Histogram Roughness</span><span class="adv-val">${fmtNum(chist.roughness, 6)}</span></li>
    <li><span>JPEG Ghost Spread</span> <span class="adv-val">${jpegNA ? 'N/A (non-JPEG)' : fmtNum(jpeg.ghost_spread, 3)}</span></li>
  `;
  container.appendChild(makeAdvCard('Advanced Signals', forensicItems));

  // Card 3 — Layer Weights
  const weightItems = `
    <li><span>Filename Weight</span>  <span class="adv-val">${fnLayer ? fnLayer.weight_pct : '0%'}</span></li>
    <li><span>Filename AI Score</span><span class="adv-val">${fnLayer ? fnLayer.ai_pts + ' pts' : '—'}</span></li>
    <li><span>Metadata Weight</span>  <span class="adv-val">${mdLayer ? mdLayer.weight_pct : '0%'}</span></li>
    <li><span>Metadata AI Score</span><span class="adv-val">${mdLayer ? mdLayer.ai_pts + ' pts' : '—'}</span></li>
    <li><span>Model & Forensic Wt</span><span class="adv-val">${mlLayer ? mlLayer.weight_pct : '0%'}</span></li>
    <li><span>Model AI Score</span>   <span class="adv-val">${mlLayer ? mlLayer.ai_pts + ' pts' : '—'}</span></li>
  `;
  container.appendChild(makeAdvCard('Layer Breakdown', weightItems));

  // Signal Log — full width
  const allSignals = [
    ...(fnLayer ? fnLayer.signals || [] : []),
    ...(mdLayer ? mdLayer.signals || [] : []),
    ...(mlLayer ? mlLayer.signals || [] : []),
  ];
  const logHtml = allSignals.length
    ? allSignals.map(s => {
        const clean = esc(s)
          .replace(/^\[AI\]\s*/i,    '<span style="color:var(--red)">⚠ </span>')
          .replace(/^\[REAL\]\s*/i,  '<span style="color:var(--green)">✓ </span>')
          .replace(/^\[INFO\]\s*/i,  '<span style="color:var(--text-3)">ℹ </span>')
          .replace(/^\[ERROR\]\s*/i, '<span style="color:var(--orange)">✕ </span>');
        return clean;
      }).join('\n')
    : 'No signals detected.';

  const logEl = document.createElement('div');
  logEl.className = 'adv-log';
  logEl.style.gridColumn = '1 / -1';
  logEl.innerHTML = `<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:var(--text-3);margin-bottom:10px;">Diagnostic Signal Log</div>${logHtml}`;
  container.appendChild(logEl);
}

function makeAdvCard(title, itemsHtml) {
  const card = document.createElement('div');
  card.className = 'adv-card';
  card.innerHTML = `<div class="adv-card-title">${title}</div><ul class="adv-list">${itemsHtml}</ul>`;
  return card;
}

// ── Toast Notification ────────────────────────────────────────────
function showToast(msg, type = 'info') {
  // Remove existing toast if any
  const old = document.getElementById('imgauth-toast');
  if (old) old.remove();

  const toast = document.createElement('div');
  toast.id = 'imgauth-toast';
  toast.style.cssText = `
    position:fixed; bottom:28px; left:50%; transform:translateX(-50%);
    background:${type === 'error' ? 'var(--red-bg)' : 'var(--surface-solid)'};
    border:1px solid ${type === 'error' ? 'var(--red-border)' : 'var(--border)'};
    color:var(--text); padding:12px 24px; border-radius:10px;
    font-size:0.875rem; font-weight:600; font-family:var(--font);
    box-shadow:var(--shadow-lg); z-index:9999;
    animation:fadeInUp 0.25s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// Add toast animation
const toastStyle = document.createElement('style');
toastStyle.textContent = `
  @keyframes fadeInUp {
    from { opacity:0; transform:translateX(-50%) translateY(12px); }
    to   { opacity:1; transform:translateX(-50%) translateY(0); }
  }
`;
document.head.appendChild(toastStyle);

// ── Utility ───────────────────────────────────────────────────────
function esc(str) {
  if (!str) return '';
  return String(str).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
}