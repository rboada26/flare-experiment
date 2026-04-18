'use strict';

/* ── Presets ─────────────────────────────────────────────────────────────── */
const PRESETS = {
  quick:    { sessions: 2,  rounds: 3,  window: 15,  minPackets: 25, trees: 50,  minSize: 0  },
  balanced: { sessions: 4,  rounds: 5,  window: 30,  minPackets: 50, trees: 100, minSize: 0  },
  full:     { sessions: 8,  rounds: 5,  window: 30,  minPackets: 50, trees: 200, minSize: 0  },
  paper:    { sessions: 16, rounds: 5,  window: 300, minPackets: 50, trees: 200, minSize: 66 },
};

const ARCH_COLORS = {
  simplecnn: '#00d4aa', resnet: '#818cf8', mobilenet: '#f59e0b',
  gru: '#f472b6', lstm: '#60a5fa', bilstm: '#34d399',
};

/* ── Chart instances ──────────────────────────────────────────────────────── */
let chartCaptures = null, chartF1Variants = null,
    chartImportance = null, chartFamily = null;

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
    if (currentPhase === 'test') refreshModelInfoCard();
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
  if (state === 'done') playDoneChime();
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
  const card   = $('#model-info-card');
  const locked = $('#locked-params');
  const btnTest = $('#btn-test');
  if (!card) return;

  if (!info || !info.exists) {
    card.innerHTML = '<div class="model-info-none">No trained model.<br>Run Train phase first.</div>';
    if (locked) locked.textContent = '—';
    if (btnTest) btnTest.disabled = true;
    return;
  }

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
  setStatus('running');
  switchTab('live');
  buildSessionSteps(params.sessions);
  $$('.phase-tab').forEach(t => t.classList.add('disabled'));
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

/* ── Test run/stop ────────────────────────────────────────────────────────── */
const btnTest     = $('#btn-test');
const btnStopTest = $('#btn-stop-test');

btnTest.addEventListener('click', async () => {
  const rounds = getParam('test-rounds') || 5;
  const archs  = getSelectedArchs('arch-toggles');

  logOutput.innerHTML = '';
  setStatus('running');
  switchTab('live');
  buildSessionSteps(1);
  $$('.phase-tab').forEach(t => t.classList.add('disabled'));
  btnTest.disabled     = true;
  btnStopTest.disabled = false;

  // Clear previous prediction
  $('#pred-grid').innerHTML    = '';
  $('#pred-overall-row').classList.add('hidden');
  $('#pred-model-row').classList.add('hidden');
  $('#pred-empty').style.display = '';

  try { await window.lab.startTest({ rounds, archs }); }
  catch (err) { appendLog(`[renderer] ${err.message}`); setStatus('error'); }

  btnTest.disabled     = false;
  btnStopTest.disabled = true;
  $$('.phase-tab').forEach(t => t.classList.remove('disabled'));
});

btnStopTest.addEventListener('click', async () => {
  btnStopTest.disabled = true;
  await window.lab.stopExperiment();
  btnTest.disabled = false;
});

/* ── IPC events ───────────────────────────────────────────────────────────── */
window.lab.onLog(appendLog);
window.lab.onStatus(setStatus);
window.lab.onSessionStart(n => { appendLog(`[session] Session ${n} started`); setStepRunning(n); });
window.lab.onSessionDone(n  => { appendLog(`[session] Session ${n} complete`); setStepDone(n);   });
window.lab.onCaptures(renderCaptureChart);
window.lab.onResults(data => { switchTab('results'); renderResults(data); });
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
Chart.defaults.color       = '#6b7799';
Chart.defaults.borderColor = '#2a2a50';
Chart.defaults.font.family = "'Menlo','Consolas',monospace";
Chart.defaults.font.size   = 11;

const TOOLTIP_DEFAULTS = {
  backgroundColor: '#1a1a3a', borderColor: '#2a2a50', borderWidth: 1,
  titleColor: '#dde2f0', bodyColor: '#dde2f0',
};
const SCALE_DEFAULTS = {
  x: { grid: { color: '#1e1e40' }, ticks: { color: '#6b7799' } },
  y: { grid: { color: '#1e1e40' }, ticks: { color: '#6b7799' } },
};

function clearCharts() {
  [chartCaptures, chartF1Variants, chartImportance, chartFamily].forEach(c => { if (c) c.destroy(); });
  chartCaptures = chartF1Variants = chartImportance = chartFamily = null;
  $('#cm-container').innerHTML = '';
}

/* ── Capture chart ────────────────────────────────────────────────────────── */
function renderCaptureChart(list) {
  const canvas = $('#chart-captures');
  if (!canvas) return;
  const labels = list.map(f => f.name.length > 28 ? '...' + f.name.slice(-25) : f.name);
  const data   = list.map(f => f.size);
  const colors = list.map(f => ARCH_COLORS[f.arch] || '#6b7799');

  if (chartCaptures) {
    chartCaptures.data.labels = labels;
    chartCaptures.data.datasets[0].data = data;
    chartCaptures.data.datasets[0].backgroundColor = colors;
    chartCaptures.update('none');
    return;
  }
  chartCaptures = new Chart(canvas, {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0, borderRadius: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { ...TOOLTIP_DEFAULTS,
        callbacks: { label: ctx => `${(ctx.raw/1024).toFixed(1)} KB` } } },
      scales: {
        x: { ...SCALE_DEFAULTS.x, ticks: { color:'#6b7799', maxRotation:45, font:{size:9} } },
        y: { ...SCALE_DEFAULTS.y, ticks: { color:'#6b7799', callback: v => (v/1024).toFixed(0)+'K' } },
      },
    },
  });
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
            datasets: [{ data: means, backgroundColor:['#00d4aa','#818cf8','#f59e0b','#f472b6'],
                         borderWidth:0, borderRadius:4 }] },
    options: { responsive:true, maintainAspectRatio:false,
      plugins: { legend:{display:false}, tooltip:{...TOOLTIP_DEFAULTS,
        callbacks:{label:ctx=>(ctx.raw*100).toFixed(2)+'% F1'}} },
      scales: { x:{...SCALE_DEFAULTS.x,ticks:{color:'#6b7799',font:{size:10}}},
                y:{...SCALE_DEFAULTS.y,ticks:{color:'#6b7799',callback:v=>(v*100).toFixed(0)+'%'},min:0,max:1} } },
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
      scales:{x:{...SCALE_DEFAULTS.x,ticks:{color:'#6b7799',callback:v=>v.toFixed(3)}},
              y:{...SCALE_DEFAULTS.y,ticks:{color:'#dde2f0',font:{size:10}}}} },
  });
}

function renderFamilyChart(fj) {
  const canvas = $('#chart-family');
  if (!canvas) return;
  if (chartFamily) chartFamily.destroy();
  const { cnn, rnn } = fj;
  const labels = [], data = [], colors = [];
  if (cnn) { labels.push('CNN Family'); data.push(cnn.mean); colors.push('#00d4aa'); }
  if (rnn) { labels.push('RNN Family'); data.push(rnn.mean); colors.push('#818cf8'); }
  if (!labels.length) return;
  chartFamily = new Chart(canvas, {
    type:'bar',
    data:{ labels, datasets:[{data, backgroundColor:colors, borderWidth:0, borderRadius:4}] },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{...TOOLTIP_DEFAULTS,callbacks:{
        label:(ctx)=>{
          const family = ctx.dataIndex === 0 ? (cnn || rnn) : rnn;
          return `${(ctx.raw*100).toFixed(1)}% ± ${(family.std*100).toFixed(1)}%`;
        }}}},
      scales:{x:{...SCALE_DEFAULTS.x,ticks:{color:'#dde2f0',font:{size:12}}},
              y:{...SCALE_DEFAULTS.y,ticks:{color:'#6b7799',callback:v=>(v*100).toFixed(0)+'%'},min:0,max:1}} },
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
      td.style.color=int>0.45?'#ffffff':'#6b7799';
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
            <span class="vote-bar-wrap"><span class="vote-bar" style="width:${pct}%;background:${ARCH_COLORS[name.toLowerCase()]||'#6b7799'}"></span></span>
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

/* ── Init: load model info on startup ─────────────────────────────────────── */
refreshModelInfoCard();
