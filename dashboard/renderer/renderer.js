'use strict';

/* ── Presets ─────────────────────────────────────────────────────────────── */
const PRESETS = {
  quick:    { sessions: 2,  rounds: 3,  window: 15,  minPackets: 25, trees: 50,  minSize: 0  },
  balanced: { sessions: 4,  rounds: 5,  window: 30,  minPackets: 50, trees: 100, minSize: 0  },
  full:     { sessions: 8,  rounds: 5,  window: 30,  minPackets: 50, trees: 200, minSize: 0  },
  paper:    { sessions: 16, rounds: 5,  window: 60,  minPackets: 50, trees: 200, minSize: 66 },
};

const ARCH_COLORS = {
  lstm:      '#60a5fa',
  gru:       '#f472b6',
  bilstm:    '#34d399',
  simplecnn: '#94a3b8',
  mobilenet: '#fb923c',
  resnet:    '#a78bfa',
};
const ARCH_FAMILY = { lstm:'r', gru:'r', bilstm:'r', simplecnn:'c', mobilenet:'c', resnet:'c' };

/* ── Chart instances ──────────────────────────────────────────────────────── */
let chartCapturesTrain = null, chartCapturesTest = null,
    chartF1Variants = null, chartImportance = null, chartFamily = null,
    chartAttackerRate = null, chartEavBox = null, chartEavSizes = null;

/* ── Attacker-view rate tracking ──────────────────────────────────────────── */
let prevCapSizes = {};   // arch → cumulative bytes at last poll
let prevCapTime  = null; // ms timestamp of last poll
let archRateMap  = {};   // arch → smoothed bytes/sec
let boxStatsMap  = {};   // arch → {min,q1,median,q3,max,mean,count}

// tracks whether the live tab showing is train or test
let livePhase = 'train';

/* ── DOM helpers ──────────────────────────────────────────────────────────── */
const $  = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));

/* ── Phase switching ──────────────────────────────────────────────────────── */
let currentPhase   = 'train';
let currentMinSize = 0;

$$('.phase-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    if (tab.classList.contains('disabled')) return;
    currentPhase = tab.dataset.phase;
    $$('.phase-tab').forEach(t => t.classList.toggle('active', t === tab));
    $$('.phase-content').forEach(c => c.classList.toggle('hidden', c.id !== `phase-${currentPhase}`));
    if (currentPhase === 'test') { refreshModelInfoCard(); }
  });
});

/* ── Pill selection ───────────────────────────────────────────────────────── */
function getParam(groupId) {
  const a = $(`#pills-${groupId} .pill.active`);
  return a ? Number(a.dataset.value) : null;
}
function setParam(groupId, value) {
  const pills = $$(`#pills-${groupId} .pill`);
  let matched = false;
  pills.forEach(p => { const ok = Number(p.dataset.value) === Number(value); p.classList.toggle('active', ok); if (ok) matched = true; });
  if (!matched && pills.length) {
    let best = pills[0], bestD = Infinity;
    pills.forEach(p => { const d = Math.abs(Number(p.dataset.value) - Number(value)); if (d < bestD) { bestD = d; best = p; } });
    best.classList.add('active');
  }
}

$$('.pill-group').forEach(g => {
  g.addEventListener('click', e => {
    const pill = e.target.closest('.pill');
    if (!pill) return;
    $$(`#${g.id} .pill`).forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
  });
});

/* ── Presets ──────────────────────────────────────────────────────────────── */
$$('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const p = PRESETS[btn.dataset.preset];
    if (!p) return;
    setParam('sessions',   p.sessions);
    setParam('rounds',     p.rounds);
    setParam('window',     p.window);
    setParam('minPackets', p.minPackets);
    setParam('trees',      p.trees);
    currentMinSize = p.minSize || 0;
  });
});

/* ── Train CV feasibility estimate ───────────────────────────────────────── */
function updateTrainCvEstimate() {
  const el = $('#train-cv-estimate');
  if (!el) return;

  const sessions = getParam('sessions')   || 2;
  const rounds   = getParam('rounds')     || 5;
  const window   = getParam('window')     || 30;

  // Lower-bound estimate: RNN rounds take ~20-30 s on the tiny dataset.
  // windows_per_round ≈ round_duration / window, conservatively 20 s / window.
  // We clamp to [0.1, 1] so large windows don't go completely to zero.
  const winPerRound = Math.min(1, Math.max(0.1, 20 / window));
  const est = Math.floor(sessions * rounds * winPerRound);

  let cls, icon, msg;
  if (est >= 10) {
    cls  = 'ok';
    icon = '✓';
    msg  = `~${est} windows/class — good for 5-fold CV`;
  } else if (est >= 5) {
    cls  = 'warn';
    icon = '⚠';
    msg  = `~${est} windows/class — marginal; add sessions`;
  } else {
    cls  = 'danger';
    icon = '✗';
    msg  = `~${est} windows/class — too few for 5-fold CV`;
  }

  el.className   = `train-cv-estimate ${cls}`;
  el.innerHTML   = `<span class="tcv-icon">${icon}</span><span class="tcv-msg">${msg}</span>`;
}

// Re-run whenever sessions, rounds, or window changes
['sessions', 'rounds', 'window'].forEach(id => {
  $(`#pills-${id}`)?.addEventListener('click', updateTrainCvEstimate);
});
// Also re-run when a preset button is clicked (they change all three at once)
$$('.preset-btn').forEach(btn => btn.addEventListener('click', updateTrainCvEstimate));

updateTrainCvEstimate();  // initialise on load

/* ── Arch toggles (multi-select, both phases) ─────────────────────────────── */
function wireArchToggleGroup(groupId) {
  const group = $(`#${groupId}`);
  if (!group) return;
  group.addEventListener('click', e => {
    const btn = e.target.closest('.arch-toggle');
    if (!btn) return;
    const active = $$(`#${groupId} .arch-toggle.active`);
    if (btn.classList.contains('active') && active.length <= 1) return;
    btn.classList.toggle('active');
  });
}
wireArchToggleGroup('train-arch-toggles');
wireArchToggleGroup('arch-toggles');

function getSelectedArchs(groupId) {
  return $$(`#${groupId} .arch-toggle.active`).map(b => b.dataset.arch);
}

/* ── Tab switching ────────────────────────────────────────────────────────── */
$$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${btn.dataset.tab}`).classList.add('active');
  });
});
function switchTab(name) {
  $$('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab-content').forEach(c => c.classList.toggle('active', c.id === `tab-${name}`));
}

/* ── Log panel ────────────────────────────────────────────────────────────── */
const logOutput = $('#log-output');

function appendLog(msg) {
  const line = document.createElement('div');
  line.className = 'log-line';
  if      (msg.includes('[stderr]'))                                               line.classList.add('stderr');
  else if (msg.includes('╔') || msg.includes('║') || msg.includes('╚'))           line.classList.add('header');
  else if (/^\[[\d/]+\]/.test(msg) || msg.startsWith('['))                        line.classList.add('step');
  else if (/complete|saved|done/i.test(msg))                                       line.classList.add('done');
  else if (/warn|skip/i.test(msg))                                                 line.classList.add('warn');
  line.textContent = msg;
  logOutput.appendChild(line);
  logOutput.scrollTop = logOutput.scrollHeight;
}
$('#btn-clear-log').addEventListener('click', () => { logOutput.innerHTML = ''; });

/* ── Done chime ───────────────────────────────────────────────────────────── */
function playDoneChime() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    // Ascending major triad: C5 → E5 → G5 → C6
    [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'sine';
      osc.frequency.value = freq;
      const t = ctx.currentTime + i * 0.13;
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(0.22, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.5);
      osc.start(t);
      osc.stop(t + 0.5);
    });
  } catch (_) {}
}

/* ── Status badges ────────────────────────────────────────────────────────── */
const LABELS = { idle:'Idle', running:'Running', done:'Done', stopped:'Stopped', error:'Error' };

function setStatus(state) {
  [$('#status-badge'), $('#status-badge-test')].forEach(el => {
    if (!el) return;
    el.className = `status-badge status-${state}`;
    el.textContent = LABELS[state] || state;
  });
  if (state === 'done') {
    playDoneChime();
    if (currentPhase === 'train') showSaveModelPrompt();
  }
}

/* ── Session steps ────────────────────────────────────────────────────────── */
function buildSessionSteps(n) {
  const list = $('#steps-list');
  list.innerHTML = '';
  for (let i = 1; i <= n; i++) {
    const row = document.createElement('div');
    row.className = 'step-row'; row.id = `step-${i}`;
    row.innerHTML = `<div class="step-icon" id="step-icon-${i}">○</div><span>Session ${i}</span>`;
    list.appendChild(row);
  }
  $('#session-steps').classList.remove('hidden');
}
function setStepRunning(n) {
  const row = $(`#step-${n}`), icon = $(`#step-icon-${n}`);
  if (row)  row.className  = 'step-row running';
  if (icon) icon.textContent = '↻';
}
function setStepDone(n) {
  const row = $(`#step-${n}`), icon = $(`#step-icon-${n}`);
  if (row)  row.className  = 'step-row done';
  if (icon) icon.textContent = '✓';
}

/* ── Model info card ──────────────────────────────────────────────────────── */
let cachedModelInfo = null;

async function refreshModelInfoCard() {
  const info = await window.lab.loadModelInfo();
  cachedModelInfo = info;
  renderModelInfoCard(info);
}

function renderModelInfoCard(info) {
  const card    = $('#model-info-card');
  const locked  = $('#locked-params');
  const btnTest = $('#btn-test');
  const btnExport = $('#btn-export-model');
  if (!card) return;

  if (!info || !info.exists) {
    card.innerHTML = '<div class="model-info-none">No trained model.<br>Run Train phase first.</div>';
    if (locked)  locked.textContent = '—';
    if (btnTest) btnTest.disabled = true;
    if (btnExport) btnExport.disabled = true;
    return;
  }
  if (btnExport) btnExport.disabled = false;

  const date = info.trainedAt ? new Date(info.trainedAt).toLocaleDateString() : '?';
  const time = info.trainedAt ? new Date(info.trainedAt).toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }) : '';
  card.innerHTML = `
    <div class="model-info-header">Model Ready</div>
    <div class="model-info-row"><span class="mi-label">Trained</span><span class="mi-val">${date} ${time}</span></div>
    <div class="model-info-row"><span class="mi-label">Sessions</span><span class="mi-val">${info.sessions ?? '?'}</span></div>
    <div class="model-info-row"><span class="mi-label">Rounds</span><span class="mi-val">${info.rounds ?? '?'}</span></div>
    <div class="model-info-row"><span class="mi-label">Window</span><span class="mi-val">${info.window ?? '?'}s</span></div>
    <div class="model-info-row"><span class="mi-label">Min Pkts</span><span class="mi-val">${info.minPackets ?? '?'}</span></div>
    <div class="model-info-row"><span class="mi-label">RF Trees</span><span class="mi-val">${info.trees ?? '?'}</span></div>
  `;

  if (locked) {
    const sizeRow = info.minSize > 0
      ? `<div class="locked-row"><span class="locked-key">Min Size</span><span class="locked-val">${info.minSize}b</span></div>`
      : '';
    locked.innerHTML = `
      <div class="locked-row"><span class="locked-key">Window</span><span class="locked-val">${info.window ?? '?'}s</span></div>
      <div class="locked-row"><span class="locked-key">Min Pkts</span><span class="locked-val">${info.minPackets ?? '?'}</span></div>
      ${sizeRow}
    `;
  }
  if (btnTest) btnTest.disabled = false;
}

/* ── Save-model prompt (auto-shown after training) ───────────────────────── */
function showSaveModelPrompt() {
  const el = $('#save-model-prompt');
  if (!el) return;
  $('#save-model-name').value = '';
  $('#save-model-status').textContent = '';
  $('#save-model-status').className = 'bin-status';
  el.classList.remove('hidden');
  setTimeout(() => $('#save-model-name')?.focus(), 80);
}

function hideSaveModelPrompt() {
  $('#save-model-prompt')?.classList.add('hidden');
}

$('#btn-save-confirm')?.addEventListener('click', async () => {
  const nameEl  = $('#save-model-name');
  const statEl  = $('#save-model-status');
  const name    = nameEl?.value.trim() || '';
  if (!name) { nameEl?.focus(); return; }

  $('#btn-save-confirm').disabled = true;
  statEl.className = 'bin-status info';
  statEl.textContent = 'Saving…';

  try {
    const res = await window.lab.exportModel({ name });
    if (res.ok) {
      statEl.className = 'bin-status ok';
      statEl.textContent = `✓  ${res.relPath}`;
    } else {
      statEl.className = 'bin-status err';
      statEl.textContent = res.error || 'Export failed.';
    }
  } catch (e) {
    statEl.className = 'bin-status err';
    statEl.textContent = e.message;
  }
  $('#btn-save-confirm').disabled = false;
});

// Allow Enter key to trigger save
$('#save-model-name')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') $('#btn-save-confirm')?.click();
  if (e.key === 'Escape') hideSaveModelPrompt();
});

$('#btn-save-skip')?.addEventListener('click', hideSaveModelPrompt);

/* ── Export / Import .bin model ──────────────────────────────────────────── */
function setBinStatus(msg, type = 'info') {
  const el = $('#bin-status');
  if (!el) return;
  el.textContent = msg;
  el.className = `bin-status ${type}`;
  if (msg) setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 6000);
}

$('#btn-export-model')?.addEventListener('click', async () => {
  const btn = $('#btn-export-model');
  btn.disabled = true;
  setBinStatus('Exporting…', 'info');
  try {
    const res = await window.lab.exportModel();
    if (res.canceled) { setBinStatus(''); }
    else if (res.ok)  { setBinStatus(`Saved: ${res.filePath.split('/').pop()}`, 'ok'); }
    else              { setBinStatus(`Error: ${res.error}`, 'err'); }
  } catch (e) {
    setBinStatus(`Error: ${e.message}`, 'err');
  }
  btn.disabled = !cachedModelInfo?.exists;
});

$('#btn-import-model')?.addEventListener('click', async () => {
  const btn = $('#btn-import-model');
  btn.disabled = true;
  setBinStatus('Opening file…', 'info');
  try {
    const res = await window.lab.importModel();
    if (res.canceled) {
      setBinStatus('');
    } else if (res.ok) {
      const h = res.header;
      setBinStatus(`Loaded: F1 ${(h.cv_f1_mean * 100).toFixed(1)}%  ${h.sessions}sess · ${h.window}s window`, 'ok');
      await refreshModelInfoCard();
    } else {
      setBinStatus(`Error: ${res.error}`, 'err');
    }
  } catch (e) {
    setBinStatus(`Error: ${e.message}`, 'err');
  }
  btn.disabled = false;
});

/* ── Train run/stop ───────────────────────────────────────────────────────── */
const btnTrain     = $('#btn-train');
const btnStopTrain = $('#btn-stop-train');

btnTrain.addEventListener('click', async () => {
  const params = {
    sessions:   getParam('sessions')   || 2,
    rounds:     getParam('rounds')     || 5,
    window:     getParam('window')     || 30,
    minPackets: getParam('minPackets') || 50,
    trees:      getParam('trees')      || 100,
    minSize:    currentMinSize,
    archs:      getSelectedArchs('train-arch-toggles'),
  };

  logOutput.innerHTML = '';
  resetMetricCards();
  clearCharts();
  hideSaveModelPrompt();
  livePhase = 'train';
  setStatus('running');
  switchTab('live-train');
  buildSessionSteps(params.sessions);
  $$('.phase-tab').forEach(t => { if (t.dataset.phase !== 'info') t.classList.add('disabled'); });
  btnTrain.disabled     = true;
  btnStopTrain.disabled = false;

  try { await window.lab.startTrain(params); }
  catch (err) { appendLog(`[renderer] ${err.message}`); setStatus('error'); }

  btnTrain.disabled     = false;
  btnStopTrain.disabled = true;
  $$('.phase-tab').forEach(t => t.classList.remove('disabled'));
  refreshModelInfoCard();
});

btnStopTrain.addEventListener('click', async () => {
  btnStopTrain.disabled = true;
  await window.lab.stopExperiment();
  btnTrain.disabled = false;
});

/* ── MIST toggle + live bandwidth estimate ────────────────────────────────── */
const mistToggle   = $('#mist-toggle');
const mistParamsEl = $('#mist-params');

function updateMistEstimate() {
  const el = $('#mist-estimate');
  if (!el) return;
  if (!mistToggle?.checked) { el.textContent = ''; el.className = 'mist-estimate'; return; }
  const pFixed   = getParam('mist-pfixed') || 262144;
  const rate     = getParam('mist-rate')   || 10;
  const numArchs = getSelectedArchs('arch-toggles').length || 6;
  const bwMBps   = (pFixed * rate * 2 * numArchs) / (1024 * 1024);
  const tenMinGB = (bwMBps * 600) / 1024;
  el.textContent = `~${bwMBps.toFixed(1)} MB/s  ·  ~${tenMinGB.toFixed(1)} GB / 10-min test`;
  el.className = 'mist-estimate ' + (tenMinGB < 50 ? 'ok' : tenMinGB < 150 ? 'warn' : 'danger');
}

if (mistToggle) {
  mistToggle.addEventListener('change', () => {
    if (mistParamsEl) mistParamsEl.classList.toggle('hidden', !mistToggle.checked);
    updateMistEstimate();
    updateAttackerView();
  });
}
// Re-run estimate whenever P_fixed, Rate, or selected archs change
['pills-mist-pfixed', 'pills-mist-rate'].forEach(id => {
  $(`#${id}`)?.addEventListener('click', updateMistEstimate);
});
$('#arch-toggles')?.addEventListener('click', updateMistEstimate);

/* ── Test run/stop ────────────────────────────────────────────────────────── */
const btnTest         = $('#btn-test');
const btnStopTest     = $('#btn-stop-test');
const btnStopSniffing = $('#btn-stop-sniffing');

btnTest.addEventListener('click', async () => {
  const rounds     = 5;   // hardcoded — attacker has no say in victim's training schedule
  const archs      = getSelectedArchs('arch-toggles');
  const mist       = mistToggle?.checked || false;
  const mistPFixed = mist ? (getParam('mist-pfixed') || 262144) : 262144;
  const mistRate   = mist ? (getParam('mist-rate')   || 10)     : 10;

  logOutput.innerHTML = '';
  livePhase = 'test';
  startNewTestRun();
  setStatus('running');
  switchTab('live-test');
  buildSessionSteps(1);
  $$('.phase-tab').forEach(t => { if (t.dataset.phase !== 'info') t.classList.add('disabled'); });
  btnTest.disabled          = true;
  btnStopTest.disabled      = false;
  if (btnStopSniffing) { btnStopSniffing.disabled = false; btnStopSniffing.textContent = '⬛ Stop & Predict'; }

  // Clear previous prediction
  $('#pred-grid').innerHTML    = '';
  $('#pred-overall-row').classList.add('hidden');
  $('#pred-model-row').classList.add('hidden');
  $('#pred-empty').style.display = '';

  try { await window.lab.startTest({ rounds, archs, mist, mistPFixed, mistRate }); }
  catch (err) { appendLog(`[renderer] ${err.message}`); setStatus('error'); }

  btnTest.disabled          = false;
  btnStopTest.disabled      = true;
  if (btnStopSniffing) btnStopSniffing.disabled = true;
  $$('.phase-tab').forEach(t => t.classList.remove('disabled'));
});

btnStopTest.addEventListener('click', async () => {
  btnStopTest.disabled = true;
  if (btnStopSniffing) btnStopSniffing.disabled = true;
  await window.lab.stopExperiment();
  btnTest.disabled = false;
});

if (btnStopSniffing) {
  btnStopSniffing.addEventListener('click', async () => {
    btnStopSniffing.disabled    = true;
    btnStopSniffing.textContent = '⬛ Classifying...';
    await window.lab.stopSniffing();
  });
}

/* ── IPC events ───────────────────────────────────────────────────────────── */
window.lab.onLog(appendLog);
window.lab.onStatus(setStatus);
window.lab.onSessionStart(n => { appendLog(`[session] Session ${n} started`); setStepRunning(n); });
window.lab.onSessionDone(n  => { appendLog(`[session] Session ${n} complete`); setStepDone(n);   });
window.lab.onCaptures(renderCaptureChart);
window.lab.onResults(data => {
  switchTab('results');
  renderResults(data);
});
window.lab.onPrediction(data => { switchTab('prediction'); renderPrediction(data); });
window.lab.onModelInfo(info => { cachedModelInfo = info; renderModelInfoCard(info); });

/* ── CSV parser ───────────────────────────────────────────────────────────── */
function parseCSV(text) {
  if (!text) return { headers: [], rows: [] };
  const lines   = text.trim().split('\n').filter(l => l.trim());
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const rows    = lines.slice(1).map(line => {
    const vals = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''));
    const obj  = {};
    headers.forEach((h, i) => { obj[h] = vals[i] ?? ''; });
    return obj;
  });
  return { headers, rows };
}

/* ── Chart defaults ───────────────────────────────────────────────────────── */
Chart.defaults.color       = '#6a6a6a';
Chart.defaults.borderColor = '#333333';
Chart.defaults.font.family = "'Menlo','Consolas',monospace";
Chart.defaults.font.size   = 11;

const TOOLTIP_DEFAULTS = {
  backgroundColor: '#212121', borderColor: '#333333', borderWidth: 1,
  titleColor: '#d4d4d4', bodyColor: '#d4d4d4',
};
const SCALE_DEFAULTS = {
  x: { grid: { color: '#252525' }, ticks: { color: '#6a6a6a' } },
  y: { grid: { color: '#252525' }, ticks: { color: '#6a6a6a' } },
};

function clearCharts() {
  [chartCapturesTrain, chartCapturesTest, chartF1Variants, chartImportance, chartFamily,
   chartAttackerRate, chartEavBox, chartEavSizes]
    .forEach(c => { if (c) c.destroy(); });
  chartCapturesTrain = chartCapturesTest = chartF1Variants = chartImportance = chartFamily =
    chartAttackerRate = chartEavBox = chartEavSizes = null;
  $('#cm-container').innerHTML = '';
  clearDumbbell();
  resetAttackerView();
  resetEavesdropper();
}

/* ── Attacker-view chart ──────────────────────────────────────────────────── */
const ARCH_ORDER_CAP = ['lstm','gru','bilstm','simplecnn','mobilenet','resnet'];

function resetAttackerView() {
  prevCapSizes = {};
  prevCapTime  = null;
  archRateMap  = {};
  boxStatsMap  = {};
  const card = $('#attacker-view-card');
  if (card) card.classList.add('hidden');
}

/* ── Eavesdropper tab ─────────────────────────────────────────────────────── */
function resetEavesdropper() {
  const chipActive = $('#eav-chip-active');
  const chipCV     = $('#eav-chip-cv');
  const chipMist   = $('#eav-chip-mist');
  if (chipActive) { chipActive.className = 'eav-chip'; chipActive.textContent = '— active'; }
  if (chipCV)     { chipCV.className     = 'eav-chip'; chipCV.textContent     = 'CV —'; }
  if (chipMist)   { chipMist.className   = 'eav-chip'; chipMist.textContent   = 'MIST OFF'; }
}

function updateEavesdropperTab(archMap) {
  if (livePhase !== 'test') return;
  const mistOn    = mistToggle?.checked || false;
  const pFixed    = getParam('mist-pfixed') || 262144;
  const ratePps   = getParam('mist-rate')   || 10;
  const expKBps   = mistOn ? (pFixed * ratePps * 2) / 1024 : null;

  // ── Chips ──
  const activeCount = ARCH_ORDER_CAP.filter(a => (archMap[a] || 0) > 0).length;
  const chipActive  = $('#eav-chip-active');
  if (chipActive) chipActive.textContent = `${activeCount}/6 active`;

  const observed = ARCH_ORDER_CAP.map(a => archRateMap[a] || 0).filter(r => r > 0);
  const chipCV   = $('#eav-chip-cv');
  if (chipCV && observed.length >= 2) {
    const mean = observed.reduce((s, v) => s + v, 0) / observed.length;
    const cv   = Math.sqrt(observed.reduce((s, v) => s + (v - mean) ** 2, 0) / observed.length) / mean;
    chipCV.textContent = `CV ${(cv * 100).toFixed(1)}%`;
    chipCV.className   = `eav-chip ${cv < 0.12 ? 'ok' : cv < 0.30 ? 'warn' : 'danger'}`;
  }

  const chipMist = $('#eav-chip-mist');
  if (chipMist) {
    chipMist.textContent = mistOn ? 'MIST ON' : 'MIST OFF';
    chipMist.className   = `eav-chip ${mistOn ? 'mist-on' : ''}`;
  }

  // Show/hide MIST params card
  const mistCard = $('#eav-mist-card');
  if (mistCard) mistCard.style.display = mistOn ? '' : 'none';

  // ── Packet size box plot ──
  const boxLabels = ARCH_ORDER_CAP.map(a => `${a.toUpperCase()} (${ARCH_FAMILY[a] || '?'})`);
  const boxCanvas = $('#chart-eav-box');
  if (boxCanvas) {
    // Invisible floating-bar dataset drives the y-scale; actual drawing is via plugin
    const scaleData = ARCH_ORDER_CAP.map(a => {
      const s = boxStatsMap[a];
      return s ? [s.min, s.max] : null;
    });

    const boxDrawPlugin = {
      id: 'boxDraw',
      afterDatasetsDraw(chart) {
        const ctx = chart.ctx;
        const yScale = chart.scales.y;
        const meta = chart.getDatasetMeta(0);
        ARCH_ORDER_CAP.forEach((arch, i) => {
          const s = boxStatsMap[arch];
          if (!s || !meta.data[i]) return;
          const color = ARCH_COLORS[arch] || '#6a6a6a';
          const cx = meta.data[i].x;
          const boxHalf = 18, capHalf = 8;
          const yMin  = yScale.getPixelForValue(s.min);
          const yMax  = yScale.getPixelForValue(s.max);
          const yQ1   = yScale.getPixelForValue(s.q1);
          const yQ3   = yScale.getPixelForValue(s.q3);
          const yMed  = yScale.getPixelForValue(s.median);
          ctx.save();
          // Whisker stem
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(cx, yMin); ctx.lineTo(cx, yMax); ctx.stroke();
          // Whisker caps
          ctx.lineWidth = 1.5;
          [[yMin, capHalf], [yMax, capHalf]].forEach(([y, hw]) => {
            ctx.beginPath(); ctx.moveTo(cx - hw, y); ctx.lineTo(cx + hw, y); ctx.stroke();
          });
          // IQR box (filled + border)
          ctx.fillStyle = color + '33';
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.fillRect(cx - boxHalf, yQ3, boxHalf * 2, yQ1 - yQ3);
          ctx.strokeRect(cx - boxHalf, yQ3, boxHalf * 2, yQ1 - yQ3);
          // Median line
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(cx - boxHalf, yMed); ctx.lineTo(cx + boxHalf, yMed); ctx.stroke();
          ctx.restore();
        });
      },
    };

    if (chartEavBox) {
      chartEavBox.data.datasets[0].data = scaleData;
      chartEavBox.update('none');
    } else {
      chartEavBox = new Chart(boxCanvas, {
        type: 'bar',
        plugins: [boxDrawPlugin],
        data: {
          labels: boxLabels,
          datasets: [{
            data: scaleData,
            backgroundColor: 'transparent',
            borderWidth: 0,
            barPercentage: 0.6,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              ...TOOLTIP_DEFAULTS,
              callbacks: {
                title: items => items[0]?.label || '',
                label: ctx => {
                  const arch = ARCH_ORDER_CAP[ctx.dataIndex];
                  const s = boxStatsMap[arch];
                  if (!s) return 'No data yet';
                  return [
                    `Median : ${s.median} B`,
                    `IQR    : ${s.q1} – ${s.q3} B`,
                    `Range  : ${s.min} – ${s.max} B`,
                    `Mean   : ${Math.round(s.mean)} B`,
                    `Packets: ${s.count.toLocaleString()}`,
                  ];
                },
              },
            },
          },
          scales: {
            x: { ...SCALE_DEFAULTS.x, ticks: { color: '#d4d4d4', font: { size: 10 } } },
            y: { ...SCALE_DEFAULTS.y, min: 0,
              ticks: { color: '#6a6a6a', callback: v => `${v} B` } },
          },
        },
      });
    }
  }

  // ── PCAP cumulative size chart ──
  const sizeData   = ARCH_ORDER_CAP.map(a => (archMap[a] || 0) / (1024 * 1024));  // MB
  const sizeCanvas = $('#chart-eav-pcap');
  if (sizeCanvas) {
    if (chartEavSizes) {
      chartEavSizes.data.datasets[0].data = sizeData;
      chartEavSizes.update('none');
    } else {
      chartEavSizes = new Chart(sizeCanvas, {
        type: 'bar',
        data: { labels: boxLabels,
          datasets: [{ label: 'PCAP (MB)', data: sizeData, backgroundColor: boxLabels.map((_, i) => ARCH_COLORS[ARCH_ORDER_CAP[i]]),
            borderWidth: 0, borderRadius: 4 }] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: { ...TOOLTIP_DEFAULTS, callbacks: {
              label: ctx => `${ctx.raw.toFixed(2)} MB`,
            }},
          },
          scales: {
            x: { ...SCALE_DEFAULTS.x, ticks: { color: '#d4d4d4', font: { size: 10 } } },
            y: { ...SCALE_DEFAULTS.y, min: 0, ticks: { color: '#6a6a6a',
              callback: v => `${v.toFixed(1)} MB` } },
          },
        },
      });
    }
  }
}

function updateAttackerView() {
  const card = $('#attacker-view-card');
  if (!card) return;

  const mistOn = mistToggle?.checked || false;
  const isTest = livePhase === 'test';
  card.classList.toggle('hidden', !(mistOn && isTest));
  if (!mistOn || !isTest) return;

  const pFixed        = getParam('mist-pfixed') || 262144;
  const ratePps       = getParam('mist-rate')   || 10;
  // Expected attacker-visible throughput per arch: P_fixed × R × 2 directions
  const expectedBps   = pFixed * ratePps * 2;
  const expectedKBps  = expectedBps / 1024;

  // ── Stat chips ──
  const pFixedEl = $('#attacker-pkt-size');
  const rateEl   = $('#attacker-wire-rate');
  const distEl   = $('#attacker-distinguishable');
  if (pFixedEl) pFixedEl.textContent = `${(pFixed / 1024).toFixed(0)} KB  (all packets uniform)`;
  if (rateEl)   rateEl.textContent   = `${(expectedKBps / 1024).toFixed(1)} MB/s  (${ratePps} pps)`;

  // Distinguishable? — coefficient of variation across observed arch rates
  const observedRates = ARCH_ORDER_CAP.map(a => archRateMap[a] || 0).filter(r => r > 0);
  if (distEl) {
    if (observedRates.length < 2) {
      distEl.textContent = 'Waiting…';
      distEl.style.color = 'var(--muted)';
    } else {
      const mean = observedRates.reduce((s, v) => s + v, 0) / observedRates.length;
      const cv   = Math.sqrt(observedRates.reduce((s, v) => s + (v - mean) ** 2, 0) / observedRates.length) / mean;
      if (cv < 0.12) {
        distEl.textContent = 'NO — uniform ✓';
        distEl.style.color = 'var(--green)';
      } else {
        distEl.textContent = 'YES — varies ✗';
        distEl.style.color = 'var(--red)';
      }
    }
  }

  // ── Rate bar chart ──
  const activeArchs = ARCH_ORDER_CAP;
  const labels  = activeArchs.map(a => `${a.toUpperCase()} (${ARCH_FAMILY[a] || '?'})`);
  const data    = activeArchs.map(a => (archRateMap[a] || 0) / 1024);   // KB/s
  const colors  = activeArchs.map(a => ARCH_COLORS[a] || '#6a6a6a');
  const target  = activeArchs.map(() => expectedKBps);

  const canvas = $('#chart-attacker-rate');
  if (!canvas) return;

  if (chartAttackerRate) {
    chartAttackerRate.data.datasets[0].data = data;
    chartAttackerRate.data.datasets[0].backgroundColor = colors;
    chartAttackerRate.data.datasets[1].data = target;
    chartAttackerRate.update('none');
    return;
  }

  chartAttackerRate = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Observed throughput',
          data,
          backgroundColor: colors,
          borderWidth: 0,
          borderRadius: 4,
          order: 2,
        },
        {
          label: `MIST target (${(expectedKBps / 1024).toFixed(1)} MB/s)`,
          data: target,
          type: 'line',
          borderColor: '#818cf8',
          borderWidth: 2,
          borderDash: [5, 4],
          pointRadius: 0,
          fill: false,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#6a6a6a', font: { size: 10 }, boxWidth: 14 },
        },
        tooltip: {
          ...TOOLTIP_DEFAULTS,
          callbacks: {
            label: ctx => ctx.datasetIndex === 0
              ? `${ctx.raw.toFixed(0)} KB/s`
              : `${ctx.raw.toFixed(0)} KB/s  (target)`,
          },
        },
      },
      scales: {
        x: { ...SCALE_DEFAULTS.x, ticks: { color: '#d4d4d4', font: { size: 10 } } },
        y: {
          ...SCALE_DEFAULTS.y,
          min: 0,
          ticks: {
            color: '#6a6a6a',
            callback: v => v >= 1024 ? `${(v / 1024).toFixed(1)} MB/s` : `${v} KB/s`,
          },
        },
      },
    },
  });
}

/* ── Capture chart ────────────────────────────────────────────────────────── */

function renderCaptureChart(list) {
  // Aggregate bytes per arch across all sessions
  const archMap = {};
  list.forEach(f => { archMap[f.arch] = (archMap[f.arch] || 0) + f.size; });

  // Merge any fresh box stats from main process
  list.forEach(f => { if (f.boxStats) boxStatsMap[f.arch] = f.boxStats; });

  // ── Compute per-arch throughput rate for attacker view ──
  if (livePhase === 'test') {
    const now = Date.now();
    if (prevCapTime !== null) {
      const dt = (now - prevCapTime) / 1000;   // seconds
      if (dt >= 1.5) {
        ARCH_ORDER_CAP.forEach(arch => {
          const cur  = archMap[arch] || 0;
          const prev = prevCapSizes[arch] || 0;
          const delta = cur - prev;
          if (delta >= 0 && cur > 0) {
            // Exponential moving average for smoother display
            const raw = delta / dt;
            archRateMap[arch] = archRateMap[arch]
              ? archRateMap[arch] * 0.4 + raw * 0.6
              : raw;
          }
        });
        prevCapTime = now;
        ARCH_ORDER_CAP.forEach(arch => { prevCapSizes[arch] = archMap[arch] || 0; });
        updateAttackerView();
        updateEavesdropperTab(archMap);
      }
    } else {
      prevCapTime = now;
      ARCH_ORDER_CAP.forEach(arch => { prevCapSizes[arch] = archMap[arch] || 0; });
    }
  }

  const activeArchs = ARCH_ORDER_CAP.filter(a => archMap[a] !== undefined);
  const labels  = activeArchs.map(a => `${a.toUpperCase()} (${ARCH_FAMILY[a] || '?'})`);
  const data    = activeArchs.map(a => archMap[a]);
  const colors  = activeArchs.map(a => ARCH_COLORS[a] || '#6a6a6a');

  const isTest  = livePhase === 'test';
  const canvasId = isTest ? '#chart-captures-test' : '#chart-captures-train';
  const canvas   = $(canvasId);
  if (!canvas) return;

  const chartRef = isTest ? chartCapturesTest : chartCapturesTrain;
  const chartOptions = {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0, borderRadius: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { ...TOOLTIP_DEFAULTS,
        callbacks: {
          title: (items) => labels[items[0].dataIndex] || '',
          label: ctx => `${(ctx.raw/1024).toFixed(1)} KB total`,
        } } },
      scales: {
        x: { ...SCALE_DEFAULTS.x, ticks: { color:'#d4d4d4', font:{ size:10 } } },
        y: { ...SCALE_DEFAULTS.y, ticks: { color:'#6a6a6a', callback: v => (v/1024).toFixed(0)+'K' } },
      },
    },
  };

  if (chartRef) {
    chartRef.data.labels = labels;
    chartRef.data.datasets[0].data   = data;
    chartRef.data.datasets[0].backgroundColor = colors;
    chartRef.update('none');
    return;
  }
  const newChart = new Chart(canvas, chartOptions);
  if (isTest) chartCapturesTest = newChart;
  else        chartCapturesTrain = newChart;
}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
function resetMetricCards() {
  ['card-bestf1','card-cnnf1','card-rnnf1','card-windows'].forEach(id => {
    const el = $(`#${id}`); if (el) el.textContent = '—';
  });
}

/* ── Train results ────────────────────────────────────────────────────────── */
function renderResults(data) {
  const { cvResults, confusionMatrix, featureImportance, familyF1, totalWindows } = data;

  const el = id => $(`#${id}`);
  if (el('card-windows')) el('card-windows').textContent = totalWindows ? totalWindows.toLocaleString() : '—';

  const cvData = parseCSV(cvResults);
  if (cvData.rows.length) {
    const vf1 = {};
    cvData.rows.forEach(r => { if (!vf1[r.variant]) vf1[r.variant] = []; vf1[r.variant].push(parseFloat(r.f1)); });
    const variants = ['flow_only','packet_only','fusion_MetaLR','fusion_MetaXGB'];
    const means = variants.map(v => { const a = vf1[v]||[]; return a.length ? a.reduce((s,x)=>s+x,0)/a.length : 0; });
    const bestF1 = Math.max(...means);
    if (el('card-bestf1')) el('card-bestf1').textContent = (bestF1*100).toFixed(1)+'%';
    renderF1VariantsChart(variants, means);
  }

  if (familyF1) {
    try {
      const fj = JSON.parse(familyF1);
      if (el('card-cnnf1')) el('card-cnnf1').textContent = fj.cnn ? (fj.cnn.mean*100).toFixed(1)+'%' : '—';
      if (el('card-rnnf1')) el('card-rnnf1').textContent = fj.rnn ? (fj.rnn.mean*100).toFixed(1)+'%' : '—';
      renderFamilyChart(fj);
    } catch (_) {}
  }

  if (confusionMatrix)   renderConfusionMatrix(confusionMatrix);
  if (featureImportance) renderImportanceChart(featureImportance);
}

function renderF1VariantsChart(variants, means) {
  const canvas = $('#chart-f1variants');
  if (!canvas) return;
  if (chartF1Variants) chartF1Variants.destroy();
  chartF1Variants = new Chart(canvas, {
    type: 'bar',
    data: { labels: variants.map(v => v.replace('fusion_','').replace('_only',' only')),
            datasets: [{ data: means, backgroundColor:[ARCH_COLORS.lstm, ARCH_COLORS.gru, ARCH_COLORS.mobilenet, ARCH_COLORS.resnet, ARCH_COLORS.simplecnn, ARCH_COLORS.bilstm].slice(0, variants.length),
                         borderWidth:0, borderRadius:4 }] },
    options: { responsive:true, maintainAspectRatio:false,
      plugins: { legend:{display:false}, tooltip:{...TOOLTIP_DEFAULTS,
        callbacks:{label:ctx=>(ctx.raw*100).toFixed(2)+'% F1'}} },
      scales: { x:{...SCALE_DEFAULTS.x,ticks:{color:'#6a6a6a',font:{size:10}}},
                y:{...SCALE_DEFAULTS.y,ticks:{color:'#6a6a6a',callback:v=>(v*100).toFixed(0)+'%'},min:0,max:1} } },
  });
}

function renderImportanceChart(csvText) {
  const canvas = $('#chart-importance');
  if (!canvas) return;
  const { rows } = parseCSV(csvText);
  const top12  = rows.slice(0,12);
  const labels = top12.map(r => r.feature||'');
  const data   = top12.map(r => parseFloat(r.importance)||0);
  if (chartImportance) chartImportance.destroy();
  chartImportance = new Chart(canvas, {
    type: 'bar',
    data: { labels, datasets:[{data, backgroundColor:'#00d4aa', borderWidth:0, borderRadius:3}] },
    options: { indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{...TOOLTIP_DEFAULTS,callbacks:{label:ctx=>ctx.raw.toFixed(4)}}},
      scales:{x:{...SCALE_DEFAULTS.x,ticks:{color:'#6a6a6a',callback:v=>v.toFixed(3)}},
              y:{...SCALE_DEFAULTS.y,ticks:{color:'#d4d4d4',font:{size:10}}}} },
  });
}

function renderFamilyChart(fj) {
  const canvas = $('#chart-family');
  if (!canvas) return;
  if (chartFamily) chartFamily.destroy();
  const { cnn, rnn } = fj;
  const ARCH_KEY = { 'SimpleCNN':'simplecnn', 'ResNet18':'resnet', 'MobileNet':'mobilenet',
                     'GRU':'gru', 'LSTM':'lstm', 'BiLSTM':'bilstm' };
  const labels = [], data = [], colors = [], stds = [];

  const addFamily = (fam, familyObj) => {
    if (!familyObj) return;
    if (familyObj.per_arch && Object.keys(familyObj.per_arch).length) {
      Object.entries(familyObj.per_arch).forEach(([name, f1]) => {
        labels.push(name); data.push(f1);
        colors.push(ARCH_COLORS[ARCH_KEY[name]] || '#6a6a6a');
        stds.push(null);
      });
    } else {
      labels.push(fam === 'cnn' ? 'CNN' : 'RNN');
      data.push(familyObj.mean);
      colors.push(fam === 'cnn' ? ARCH_COLORS.mobilenet : ARCH_COLORS.lstm);
      stds.push(familyObj.std);
    }
  };
  addFamily('cnn', cnn);
  addFamily('rnn', rnn);
  if (!labels.length) return;

  chartFamily = new Chart(canvas, {
    type:'bar',
    data:{ labels, datasets:[{data, backgroundColor:colors, borderWidth:0, borderRadius:4}] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{...TOOLTIP_DEFAULTS,callbacks:{
        label:(ctx)=>{
          const std = stds[ctx.dataIndex];
          return std != null
            ? `${(ctx.raw*100).toFixed(1)}% ± ${(std*100).toFixed(1)}%`
            : `${(ctx.raw*100).toFixed(1)}% F1`;
        }}}},
      scales:{x:{...SCALE_DEFAULTS.x,ticks:{color:'#d4d4d4',font:{size:10}}},
              y:{...SCALE_DEFAULTS.y,ticks:{color:'#6a6a6a',callback:v=>(v*100).toFixed(0)+'%'},min:0,max:1}} },
  });
}

function renderConfusionMatrix(csvText) {
  const container = $('#cm-container');
  if (!container) return;
  const lines = csvText.trim().split('\n').filter(l=>l.trim());
  if (lines.length < 2) { container.innerHTML='<p style="color:var(--muted)">No data</p>'; return; }
  const headerCols = lines[0].split(',').map(h=>h.trim().replace(/^"|"$/g,''));
  const classNames = headerCols.slice(1);
  const dataRows   = lines.slice(1).map(line => {
    const cols = line.split(',').map(c=>c.trim().replace(/^"|"$/g,''));
    return { label: cols[0], values: cols.slice(1).map(Number) };
  });
  let maxVal = 0;
  dataRows.forEach(r => r.values.forEach(v => { if(v>maxVal) maxVal=v; }));

  const table = document.createElement('table');
  table.className = 'cm-table';
  const thead = document.createElement('thead');
  const hrow  = document.createElement('tr');
  const cornerTh = document.createElement('th');
  cornerTh.className='row-label'; cornerTh.textContent='↓ pred / true →'; cornerTh.style.fontSize='8.5px';
  hrow.appendChild(cornerTh);
  classNames.forEach(n => { const th=document.createElement('th'); th.textContent=n; hrow.appendChild(th); });
  thead.appendChild(hrow); table.appendChild(thead);

  const tbody = document.createElement('tbody');
  dataRows.forEach((row,ri) => {
    const tr = document.createElement('tr');
    const th = document.createElement('td');
    th.className='row-header'; th.textContent=row.label; tr.appendChild(th);
    row.values.forEach((val,ci) => {
      const td  = document.createElement('td');
      const int = maxVal>0 ? val/maxVal : 0;
      td.style.backgroundColor=`rgba(0,180,216,${(int*0.85).toFixed(2)})`;
      td.style.color=int>0.45?'#ffffff':'#6a6a6a';
      if(ri===ci) td.classList.add('diagonal');
      td.textContent=val; td.title=`${row.label} → ${classNames[ci]}: ${val}`;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  const scroll = document.createElement('div');
  scroll.className='cm-scroll'; scroll.appendChild(table);
  container.innerHTML=''; container.appendChild(scroll);
}

/* ── Prediction results ───────────────────────────────────────────────────── */
const ARCH_DISPLAY = {
  simplecnn:'SimpleCNN', resnet:'ResNet18', mobilenet:'MobileNet',
  gru:'GRU', lstm:'LSTM', bilstm:'BiLSTM',
};

function renderPrediction(data) {
  const { per_arch, overall, model_info } = data;
  const grid    = $('#pred-grid');
  const empty   = $('#pred-empty');
  const modelRow   = $('#pred-model-row');
  const overallRow = $('#pred-overall-row');

  if (empty) empty.style.display = 'none';
  grid.innerHTML = '';

  // Model info row
  if (modelRow && model_info) {
    modelRow.classList.remove('hidden');
    const f1 = model_info.cv_f1_mean ? (model_info.cv_f1_mean*100).toFixed(1)+'%' : '?';
    modelRow.innerHTML = `
      <span class="pm-label">Variant used:</span>
      <span class="pm-val">${model_info.best_variant || '?'}</span>
      <span class="pm-sep">·</span>
      <span class="pm-label">Train F1:</span>
      <span class="pm-val">${f1}</span>
    `;
  }

  // Overall row
  if (overallRow && overall) {
    overallRow.classList.remove('hidden');
    const pct = overall.total > 0 ? Math.round(overall.accuracy * 100) : 0;
    const color = pct >= 80 ? 'var(--green)' : pct >= 50 ? 'var(--yellow)' : 'var(--red)';
    overallRow.innerHTML = `
      <span style="color:${color};font-weight:700;font-size:15px">${overall.correct}/${overall.total} correct (${pct}%)</span>
      <span class="pm-sep" style="margin-left:8px">on fresh data</span>
    `;
  }

  // Prediction cards — only render archs present in this result
  const ARCH_ORDER = ['simplecnn','resnet','mobilenet','gru','lstm','bilstm'];
  const archs = ARCH_ORDER.filter(a => per_arch[a] !== undefined);
  archs.forEach(arch => {
    const r = per_arch[arch];
    const card = document.createElement('div');
    card.className = 'pred-card' + (r ? (r.correct ? ' correct' : ' wrong') : ' missing');

    const displayTrue = ARCH_DISPLAY[arch] || arch;

    if (!r) {
      card.innerHTML = `
        <div class="pred-arch-name">${displayTrue}</div>
        <div class="pred-no-data">No data</div>
      `;
    } else {
      const conf     = Math.round(r.confidence * 100);
      const confColor = conf >= 80 ? 'var(--green)' : conf >= 50 ? 'var(--yellow)' : 'var(--red)';
      const verdict  = r.correct ? '✓' : '✗';
      const verdictColor = r.correct ? 'var(--green)' : 'var(--red)';
      const displayPred = ARCH_DISPLAY[r.predicted.toLowerCase()] || r.predicted;

      // Votes breakdown
      const votesHtml = Object.entries(r.votes)
        .sort((a,b) => b[1]-a[1])
        .map(([name, count]) => {
          const pct = Math.round(count / r.n_windows * 100);
          const disp = ARCH_DISPLAY[name.toLowerCase()] || name;
          return `<div class="vote-row">
            <span class="vote-name">${disp}</span>
            <span class="vote-bar-wrap"><span class="vote-bar" style="width:${pct}%;background:${ARCH_COLORS[name.toLowerCase()]||'#6a6a6a'}"></span></span>
            <span class="vote-pct">${pct}%</span>
          </div>`;
        }).join('');

      card.innerHTML = `
        <div class="pred-card-header">
          <span class="pred-arch-name" style="color:${ARCH_COLORS[arch]||'var(--accent)'}">${displayTrue}</span>
          <span class="pred-verdict" style="color:${verdictColor}">${verdict}</span>
        </div>
        <div class="pred-result-label">Predicted</div>
        <div class="pred-result-name">${displayPred}</div>
        <div class="pred-conf" style="color:${confColor}">${conf}% agreement · ${r.n_windows} windows</div>
        <div class="pred-votes">${votesHtml}</div>
      `;
    }
    grid.appendChild(card);
  });
}

/* ── Dumbbell chart (shared) ──────────────────────────────────────────────── */
const ARCH_ORDER_DB = ['lstm','gru','bilstm','simplecnn','mobilenet','resnet'];

// Tooltip for dumbbell canvas hit detection
const _dbHitMap = new WeakMap();
let   _dbTooltip = null;

function getDbTooltip() {
  if (!_dbTooltip) {
    _dbTooltip = document.createElement('div');
    Object.assign(_dbTooltip.style, {
      position:'fixed', display:'none', pointerEvents:'none',
      background:'#212121', border:'1px solid #2a2a50', borderRadius:'6px',
      padding:'6px 10px', font:'11px Menlo,Consolas,monospace', color:'#d4d4d4',
      zIndex:'9999', whiteSpace:'nowrap', lineHeight:'1.6',
    });
    document.body.appendChild(_dbTooltip);
  }
  return _dbTooltip;
}

function attachDumbbellTooltip(canvasEl) {
  if (!canvasEl || canvasEl._dbBound) return;
  canvasEl._dbBound = true;
  const tt = getDbTooltip();

  canvasEl.addEventListener('mousemove', e => {
    const pts  = _dbHitMap.get(canvasEl) || [];
    const rect = canvasEl.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let closest = null, minD = 14;
    pts.forEach(p => { const d = Math.hypot(p.x - mx, p.y - my); if (d < minD) { minD = d; closest = p; } });
    if (closest) {
      const { arch, round, ts, bytes } = closest;
      const fam  = ARCH_FAMILY[arch] ? ` (${ARCH_FAMILY[arch]})` : '';
      const timeStr = ts >= 60000 ? `${(ts/60000).toFixed(2)}m` : ts >= 1000 ? `${(ts/1000).toFixed(2)}s` : `${Math.round(ts)}ms`;
      const kbStr   = bytes >= 1048576 ? `${(bytes/1048576).toFixed(1)} MB` : `${(bytes/1024).toFixed(1)} KB`;
      tt.innerHTML  = `<strong style="color:${ARCH_COLORS[arch]}">${arch.toUpperCase()}${fam}</strong><br>Round ${round} &middot; ${timeStr}<br>${kbStr}`;
      tt.style.display = 'block';
      tt.style.left    = (e.clientX + 14) + 'px';
      tt.style.top     = (e.clientY - 10) + 'px';
    } else {
      tt.style.display = 'none';
    }
  });

  canvasEl.addEventListener('mouseleave', () => { tt.style.display = 'none'; });
}

function drawDumbbellOnCanvas(canvasEl, sessionData) {
  if (!canvasEl || !sessionData) return;

  const wrap = canvasEl.parentElement;
  const dpr  = window.devicePixelRatio || 1;
  const W    = wrap.clientWidth  || 600;
  const H    = wrap.clientHeight || 300;
  canvasEl.width        = W * dpr;
  canvasEl.height       = H * dpr;
  canvasEl.style.width  = W + 'px';
  canvasEl.style.height = H + 'px';

  const ctx = canvasEl.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const allPoints = [];
  let maxTs = 1000, maxBytes = 1;
  const numRounds = sessionData.maxRound || 1;

  ARCH_ORDER_DB.forEach(arch => {
    const archRounds = sessionData.rounds[arch];
    if (!archRounds) return;
    Object.entries(archRounds).forEach(([r, { ts, bytes }]) => {
      allPoints.push({ arch, round: parseInt(r, 10), ts, bytes });
      if (ts    > maxTs)    maxTs    = ts;
      if (bytes > maxBytes) maxBytes = bytes;
    });
  });

  const pad      = { top: 30, right: 55, bottom: 52, left: 90 };
  const pw       = W - pad.left - pad.right;
  const ph       = H - pad.top  - pad.bottom;
  const xScale   = t => pad.left + (t / maxTs) * pw;
  const yScale   = r => pad.top  + ph - ((r - 0.5) / numRounds) * ph;
  const MAX_DASH = Math.min(80, pw * 0.15);  // max total dash length

  ctx.font = '10px Menlo, Consolas, monospace';

  // Round gridlines + labels
  for (let r = 1; r <= numRounds; r++) {
    const y = yScale(r);
    ctx.strokeStyle = '#252525'; ctx.lineWidth = 1; ctx.setLineDash([3, 6]);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#6a6a6a'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`Round ${r}`, pad.left - 8, y);
  }

  // X axis
  ctx.strokeStyle = '#333333'; ctx.lineWidth = 1; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top + ph); ctx.lineTo(pad.left + pw, pad.top + ph); ctx.stroke();

  // X ticks
  ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = '#6a6a6a';
  for (let i = 0; i <= 5; i++) {
    const t   = (maxTs / 5) * i;
    const x   = xScale(t);
    const lbl = t >= 60000 ? `${(t/60000).toFixed(1)}m` : t >= 1000 ? `${(t/1000).toFixed(1)}s` : `${Math.round(t)}ms`;
    ctx.fillText(lbl, x, pad.top + ph + 7);
    ctx.strokeStyle = '#333333'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, pad.top + ph); ctx.lineTo(x, pad.top + ph + 4); ctx.stroke();
  }

  // X axis title
  ctx.fillStyle = '#6a6a6a'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  ctx.font = '9px Menlo, Consolas, monospace';
  ctx.fillText('← Time from session start (dot = round completion)', pad.left + pw / 2, H - 2);
  ctx.font = '10px Menlo, Consolas, monospace';

  // Y axis line
  ctx.strokeStyle = '#333333'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + ph); ctx.stroke();

  if (!allPoints.length) {
    ctx.fillStyle = '#6a6a6a'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('Waiting for round data…', W / 2, H / 2);
    return;
  }

  // Dashes — dot at RIGHT tip = completion time, dash extends leftward ∝ bytes
  const hitPoints = [];
  allPoints.forEach(({ arch, round, ts, bytes }) => {
    const archIdx = ARCH_ORDER_DB.indexOf(arch);
    const spread  = (archIdx - (ARCH_ORDER_DB.length - 1) / 2) * 8;
    const x       = xScale(ts);
    const y       = yScale(round) + spread;
    const dashLen = maxBytes > 0 ? Math.max(8, (bytes / maxBytes) * MAX_DASH) : 8;
    const color   = ARCH_COLORS[arch] || '#888';

    ctx.strokeStyle = color; ctx.lineWidth = 3.5; ctx.lineCap = 'round'; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(x - dashLen, y); ctx.lineTo(x, y); ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();

    hitPoints.push({ x, y, arch, round, ts, bytes });
  });
  _dbHitMap.set(canvasEl, hitPoints);

  const kbMax = (maxBytes / 1024).toFixed(0);
  ctx.fillStyle = '#3a3a60'; ctx.font = '9px Menlo, Consolas, monospace';
  ctx.textAlign = 'right'; ctx.textBaseline = 'top';
  ctx.fillText(`max dash = ${kbMax} KB`, pad.left + pw, pad.top - 2);
}

/* ── Train dumbbell ───────────────────────────────────────────────────────── */
let dumbbellSessions   = {};
let activeDumbbellSess = null;

function clearDumbbell() {
  dumbbellSessions   = {};
  activeDumbbellSess = null;
  const bar = $('#dumbbell-session-bar');
  if (bar) bar.innerHTML = '';
  const canvas = $('#dumbbell-canvas');
  if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  const empty = $('#dumbbell-empty');
  if (empty) empty.style.display = '';
}

function onTrainRoundEvent({ session, arch, round, timestamp, bytes }) {
  if (!dumbbellSessions[session]) {
    dumbbellSessions[session] = { rounds: {}, maxRound: 0 };
    addTrainSessionTab(session);
    if (activeDumbbellSess === null) switchTrainSession(session);
  }
  const s = dumbbellSessions[session];
  if (!s.rounds[arch]) s.rounds[arch] = {};
  s.rounds[arch][round] = { ts: timestamp, bytes };
  if (round > s.maxRound) s.maxRound = round;
  const empty = $('#dumbbell-empty');
  if (empty) empty.style.display = 'none';
  if (activeDumbbellSess === session) drawDumbbell();
}

function addTrainSessionTab(sessNum) {
  const bar = $('#dumbbell-session-bar');
  if (!bar || $(`#dsess-btn-${sessNum}`)) return;
  const btn = document.createElement('button');
  btn.className   = 'session-subtab';
  btn.id          = `dsess-btn-${sessNum}`;
  btn.textContent = `Session ${sessNum}`;
  btn.addEventListener('click', () => switchTrainSession(sessNum));
  bar.appendChild(btn);
}

function switchTrainSession(sessNum) {
  activeDumbbellSess = sessNum;
  $$('#dumbbell-session-bar .session-subtab').forEach(b =>
    b.classList.toggle('active', b.id === `dsess-btn-${sessNum}`)
  );
  drawDumbbell();
}

function drawDumbbell() {
  const c = $('#dumbbell-canvas');
  drawDumbbellOnCanvas(c, activeDumbbellSess !== null ? dumbbellSessions[activeDumbbellSess] : null);
  attachDumbbellTooltip(c);
}

/* ── Test dumbbell ────────────────────────────────────────────────────────── */
let testDumbbellRuns  = {};  // persists across test runs — never cleared
let activeTestRun     = null;
let testRunCounter    = 0;
let currentTestRunNum = null;

function startNewTestRun() {
  testRunCounter++;
  currentTestRunNum = testRunCounter;
  testDumbbellRuns[currentTestRunNum] = { rounds: {}, maxRound: 0 };
  addTestRunTab(currentTestRunNum);
  switchTestRun(currentTestRunNum);
  const empty = $('#test-dumbbell-empty');
  if (empty) empty.style.display = 'none';
  // Reset rate tracking for fresh eavesdropper/attacker views
  prevCapSizes = {};
  prevCapTime  = null;
  archRateMap  = {};
  updateAttackerView();
  resetEavesdropper();
}

function onTestRoundEvent({ arch, round, timestamp, bytes }) {
  if (!currentTestRunNum) return;
  const run = testDumbbellRuns[currentTestRunNum];
  if (!run) return;
  if (!run.rounds[arch]) run.rounds[arch] = {};
  run.rounds[arch][round] = { ts: timestamp, bytes };
  if (round > run.maxRound) run.maxRound = round;
  if (activeTestRun === currentTestRunNum) drawTestDumbbell();
}

function addTestRunTab(runNum) {
  const bar = $('#test-dumbbell-run-bar');
  if (!bar || $(`#trun-btn-${runNum}`)) return;
  const btn = document.createElement('button');
  btn.className   = 'session-subtab';
  btn.id          = `trun-btn-${runNum}`;
  btn.textContent = `Run ${runNum}`;
  btn.addEventListener('click', () => switchTestRun(runNum));
  bar.appendChild(btn);
}

function switchTestRun(runNum) {
  activeTestRun = runNum;
  $$('#test-dumbbell-run-bar .session-subtab').forEach(b =>
    b.classList.toggle('active', b.id === `trun-btn-${runNum}`)
  );
  drawTestDumbbell();
}

function drawTestDumbbell() {
  const c = $('#test-dumbbell-canvas');
  drawDumbbellOnCanvas(c, activeTestRun !== null ? testDumbbellRuns[activeTestRun] : null);
  attachDumbbellTooltip(c);
}

/* ── Round event router ───────────────────────────────────────────────────── */
window.lab.onRoundEvent(({ session, arch, round, timestamp, bytes, phase }) => {
  if (phase === 'test') onTestRoundEvent({ arch, round, timestamp, bytes });
  else                  onTrainRoundEvent({ session, arch, round, timestamp, bytes });
});

/* ── Init: load model info on startup ─────────────────────────────────────── */
refreshModelInfoCard();
