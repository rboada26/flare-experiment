const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('lab', {
  // ── Actions ──────────────────────────────────────────────────────────────
  startTrain:     (params) => ipcRenderer.invoke('start-train',     params),
  startTest:      (params) => ipcRenderer.invoke('start-test',      params),
  stopExperiment: ()       => ipcRenderer.invoke('stop-experiment'),
  stopSniffing:   ()       => ipcRenderer.invoke('stop-sniffing'),
  loadModelInfo:  ()       => ipcRenderer.invoke('load-model-info'),
  exportModel:    (opts)   => ipcRenderer.invoke('export-model', opts),
  importModel:    ()       => ipcRenderer.invoke('import-model'),
  // ── Events ────────────────────────────────────────────────────────────────
  onLog:          (cb) => ipcRenderer.on('log',                (_e, d) => cb(d)),
  onStatus:       (cb) => ipcRenderer.on('status',             (_e, d) => cb(d)),
  onSessionStart: (cb) => ipcRenderer.on('session-start',      (_e, d) => cb(d)),
  onSessionDone:  (cb) => ipcRenderer.on('session-done',       (_e, d) => cb(d)),
  onCaptures:     (cb) => ipcRenderer.on('capture-stats',      (_e, d) => cb(d)),
  onResults:      (cb) => ipcRenderer.on('experiment-results', (_e, d) => cb(d)),
  onPrediction:   (cb) => ipcRenderer.on('prediction-results', (_e, d) => cb(d)),
  onModelInfo:    (cb) => ipcRenderer.on('model-info',         (_e, d) => cb(d)),
  onRoundEvent:   (cb) => ipcRenderer.on('round-event',        (_e, d) => cb(d)),
});
