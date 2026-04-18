'use strict';

const { app, BrowserWindow, ipcMain } = require('electron');
const path  = require('path');
const fs    = require('fs');
const { spawn, execSync } = require('child_process');

const PARTNER_DIR  = path.join(__dirname, '..');
const CAPTURES_DIR = path.join(PARTNER_DIR, 'captures');
const MODEL_DIR    = path.join(PARTNER_DIR, 'model');

let mainWindow       = null;
let activeProc       = null;
let capturePoller    = null;
let experimentStopped = false;

// ── Window ────────────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400, height: 900, minWidth: 1100, minHeight: 700,
    backgroundColor: '#0d0d1a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    titleBarStyle: 'hiddenInset',
    title: 'PARTNER-LAB Dashboard',
  });
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { cleanupAndQuit(); });
app.on('before-quit',       () => { cleanupAndQuit(); });

function cleanupAndQuit() {
  stopCapPoller();
  if (activeProc) { try { activeProc.kill('SIGTERM'); } catch (_) {} activeProc = null; }
  try {
    const caps = execSync("docker ps --filter 'name=cap_' --format '{{.Names}}'",
      { cwd: PARTNER_DIR, timeout: 5000 }).toString().trim();
    if (caps) execSync(`docker stop ${caps.split('\n').join(' ')}`,
      { stdio: 'ignore', timeout: 10000 });
  } catch (_) {}
  try { execSync('docker compose down --remove-orphans',
    { cwd: PARTNER_DIR, stdio: 'ignore', timeout: 15000 }); } catch (_) {}
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function send(channel, data) {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(channel, data);
}
function log(msg)     { send('log', msg); console.log('[main]', msg); }
function setStatus(s) { send('status', s); }

// ── Capture poller ────────────────────────────────────────────────────────────
function startCapPoller() {
  capturePoller = setInterval(() => {
    if (!fs.existsSync(CAPTURES_DIR)) return;
    let files;
    try { files = fs.readdirSync(CAPTURES_DIR).filter(f => f.endsWith('.pcap')); }
    catch (_) { return; }
    const stats = files.map(f => {
      let size = 0;
      try { size = fs.statSync(path.join(CAPTURES_DIR, f)).size; } catch (_) {}
      const m = f.match(/session\d+_([a-z]+)_/);
      return { name: f, arch: m ? m[1] : 'unknown', size };
    });
    send('capture-stats', stats);
  }, 2000);
}
function stopCapPoller() {
  if (capturePoller) { clearInterval(capturePoller); capturePoller = null; }
}

// ── Spawn helper ──────────────────────────────────────────────────────────────
function spawnLogged(cmd, args, onLine = null) {
  return new Promise((resolve, reject) => {
    log(`$ ${cmd} ${args.join(' ')}`);
    const proc = spawn(cmd, args, { cwd: PARTNER_DIR, stdio: ['ignore', 'pipe', 'pipe'] });
    activeProc = proc;
    proc.stdout.on('data', d => {
      d.toString().split('\n').forEach(line => {
        if (!line.trim()) return;
        log(line);
        if (onLine) onLine(line);
      });
    });
    proc.stderr.on('data', d => {
      d.toString().split('\n').forEach(line => {
        if (line.trim()) log('[stderr] ' + line);
      });
    });
    proc.on('close', code => {
      activeProc = null;
      (code === 0 || experimentStopped) ? resolve(code) : reject(new Error(`${cmd} exited with code ${code}`));
    });
    proc.on('error', err => { activeProc = null; reject(err); });
  });
}

// ── Compose down helper ───────────────────────────────────────────────────────
function composeDown() {
  try { execSync('docker compose down --remove-orphans',
    { cwd: PARTNER_DIR, stdio: 'ignore', timeout: 20000 }); } catch (_) {}
}

// ── Clean captures dir ────────────────────────────────────────────────────────
function cleanCaptures() {
  try { fs.rmSync(CAPTURES_DIR, { recursive: true, force: true }); } catch (_) {}
  fs.mkdirSync(CAPTURES_DIR, { recursive: true });
}

// ── Read file helper ──────────────────────────────────────────────────────────
function readFile(name) {
  try { return fs.readFileSync(path.join(PARTNER_DIR, name), 'utf8'); } catch (_) { return null; }
}

// ── Model info ────────────────────────────────────────────────────────────────
function readModelInfo() {
  if (!fs.existsSync(path.join(MODEL_DIR, 'model.pkl'))) return { exists: false };
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'config.json'), 'utf8'));
    return { exists: true, ...cfg };
  } catch (_) { return { exists: true }; }
}

ipcMain.handle('load-model-info', () => readModelInfo());

// ── Collect + extract (shared by train and test) ──────────────────────────────
async function collectAndExtract(sessions, rounds, window, minPackets, minSize = 0, archs = null) {
  startCapPoller();
  const archArgs = archs && archs.length < 6 ? ['--archs', archs.join(',')] : [];
  try {
    await spawnLogged('bash',
      ['collect_data.sh', '--sessions', String(sessions), '--rounds', String(rounds), ...archArgs],
      line => {
        const sm = line.match(/Starting session\s+(\d+)/);
        if (sm) send('session-start', parseInt(sm[1], 10));
        const dm = line.match(/\[session\s+(\d+)\]\s+Done/);
        if (dm) send('session-done', parseInt(dm[1], 10));
      });
  } catch (err) {
    if (!experimentStopped) log(`  collect_data.sh error: ${err.message}`);
  }
  stopCapPoller();
  if (experimentStopped) return false;

  const sizeArgs = minSize > 0 ? ['--min-size', String(minSize)] : [];
  try {
    await spawnLogged('python3',
      ['extract_features.py', '--window', String(window), '--min-packets', String(minPackets), ...sizeArgs]);
  } catch (err) {
    if (!experimentStopped) { log(`  extract_features.py error: ${err.message}`); return false; }
  }
  return !experimentStopped;
}

// ── IPC: stop-experiment ──────────────────────────────────────────────────────
ipcMain.handle('stop-experiment', async () => {
  experimentStopped = true;
  stopCapPoller();
  if (activeProc) { try { activeProc.kill('SIGTERM'); } catch (_) {} }
  setStatus('stopped');
  log('--- Stopped by user ---');
  composeDown();
});

// ── IPC: start-train ──────────────────────────────────────────────────────────
ipcMain.handle('start-train', async (_e, params) => {
  const { sessions = 2, rounds = 5, window = 30, minPackets = 50, trees = 100, minSize = 0, archs } = params;
  experimentStopped = false;

  log('╔══════════════════════════════════════════════════════════╗');
  log('║            PARTNER-LAB  ·  Train Phase                  ║');
  log('╠══════════════════════════════════════════════════════════╣');
  log(`║  Sessions   : ${String(sessions).padEnd(42)}║`);
  log(`║  FL Rounds  : ${String(rounds).padEnd(42)}║`);
  log(`║  Window     : ${String(window + 's').padEnd(42)}║`);
  log(`║  Min Packets: ${String(minPackets).padEnd(42)}║`);
  log(`║  RF Trees   : ${String(trees).padEnd(42)}║`);
  log(`║  Min PktSize: ${String(minSize > 0 ? minSize + ' bytes (paper filter)' : 'none').padEnd(42)}║`);
  log('╚══════════════════════════════════════════════════════════╝');
  setStatus('running');

  // Clean everything including model
  log('\n[1/4] Cleaning artifacts...');
  ['features.csv','cv_results_multiclass.csv','confusion_matrix.csv',
   'feature_importance_multiclass.csv','family_f1.json','prediction_results.json']
    .forEach(f => { try { fs.unlinkSync(path.join(PARTNER_DIR, f)); } catch (_) {} });
  try { fs.rmSync(MODEL_DIR, { recursive: true, force: true }); } catch (_) {}
  cleanCaptures();
  log('  Done.');

  log('\n[2/4] docker compose down...');
  composeDown();
  if (experimentStopped) { setStatus('stopped'); return; }

  log(`\n[3/4] Collecting ${sessions} session(s)...`);
  const ok = await collectAndExtract(sessions, rounds, window, minPackets, minSize, archs);
  if (!ok) { composeDown(); setStatus(experimentStopped ? 'stopped' : 'error'); return; }

  log(`\n[4/4] Classifying (RF trees=${trees})...`);
  try {
    await spawnLogged('python3', ['classify.py', '--n-estimators', String(trees)]);
  } catch (err) {
    if (!experimentStopped) { log(`  classify.py error: ${err.message}`); setStatus('error'); return; }
  }
  if (experimentStopped) { composeDown(); setStatus('stopped'); return; }

  composeDown();

  // Write model config (window/minPackets needed by test phase)
  try {
    fs.mkdirSync(MODEL_DIR, { recursive: true });
    fs.writeFileSync(path.join(MODEL_DIR, 'config.json'), JSON.stringify({
      trainedAt: new Date().toISOString(),
      window, minPackets, minSize, sessions, rounds, trees,
    }, null, 2));
  } catch (_) {}

  // Send train results
  const featCsv = readFile('features.csv');
  send('experiment-results', {
    cvResults:         readFile('cv_results_multiclass.csv'),
    confusionMatrix:   readFile('confusion_matrix.csv'),
    featureImportance: readFile('feature_importance_multiclass.csv'),
    familyF1:          readFile('family_f1.json'),
    totalWindows: featCsv ? Math.max(0, featCsv.trim().split('\n').length - 1) : 0,
    params: { sessions, rounds, window, minPackets, trees },
  });
  // Also send updated model info so Test tab refreshes
  send('model-info', readModelInfo());

  log('\n╔══════════════════════════════════════════════════════════╗');
  log('║         Training complete — model saved.                 ║');
  log('╚══════════════════════════════════════════════════════════╝');
  setStatus('done');
});

// ── IPC: start-test ───────────────────────────────────────────────────────────
ipcMain.handle('start-test', async (_e, params) => {
  const { rounds = 5, archs } = params;
  experimentStopped = false;

  const modelInfo = readModelInfo();
  if (!modelInfo.exists) {
    log('ERROR: No trained model found. Train a model first.');
    setStatus('error');
    return;
  }

  const { window, minPackets, minSize = 0 } = modelInfo;

  log('╔══════════════════════════════════════════════════════════╗');
  log('║            PARTNER-LAB  ·  Test Phase                   ║');
  log('╠══════════════════════════════════════════════════════════╣');
  log(`║  FL Rounds  : ${String(rounds).padEnd(42)}║`);
  log(`║  Window     : ${String(window + 's  (locked — from training)').padEnd(42)}║`);
  log(`║  Min Packets: ${String(minPackets + '  (locked — from training)').padEnd(42)}║`);
  log(`║  Min PktSize: ${String(minSize > 0 ? minSize + ' bytes (locked — from training)' : 'none').padEnd(42)}║`);
  log('╚══════════════════════════════════════════════════════════╝');
  setStatus('running');

  // Clean test artifacts only — keep model/
  log('\n[1/3] Cleaning test artifacts...');
  ['features.csv', 'prediction_results.json']
    .forEach(f => { try { fs.unlinkSync(path.join(PARTNER_DIR, f)); } catch (_) {} });
  cleanCaptures();
  log('  Done.');

  log('\n[2/3] Collecting 1 test session...');
  composeDown();
  if (experimentStopped) { setStatus('stopped'); return; }

  const ok = await collectAndExtract(1, rounds, window, minPackets, minSize);
  if (!ok) { composeDown(); setStatus(experimentStopped ? 'stopped' : 'error'); return; }

  log('\n[3/3] Running predict.py...');
  const archArgs = archs && archs.length < 6 ? ['--archs', archs.join(',')] : [];
  if (archArgs.length) log(`  Evaluating: ${archs.join(', ')}`);
  try {
    await spawnLogged('python3', ['predict.py', ...archArgs]);
  } catch (err) {
    if (!experimentStopped) { log(`  predict.py error: ${err.message}`); setStatus('error'); return; }
  }
  if (experimentStopped) { composeDown(); setStatus('stopped'); return; }

  composeDown();

  const predRaw = readFile('prediction_results.json');
  if (predRaw) {
    send('prediction-results', JSON.parse(predRaw));
  } else {
    log('  WARNING: prediction_results.json not found.');
  }

  log('\n╔══════════════════════════════════════════════════════════╗');
  log('║              Test phase complete.                        ║');
  log('╚══════════════════════════════════════════════════════════╝');
  setStatus('done');
});
