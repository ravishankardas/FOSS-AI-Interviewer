"use strict";

const SAMPLE_RATE = 16000;
const CHUNK = 512; // samples per WS frame, matches VAD chunk_size

// --- DOM ---
const $ = (id) => document.getElementById(id);
const setupEl = $("setup");
const interviewEl = $("interview");
const reportEl = $("report");
const onairEl = $("onair") || document.querySelector(".onair");
const lampEl = $("lamp");
const waveEl = $("wave");
const onairLabel = $("onair-label");
const speakerTag = $("speaker-tag");
const viewReportBtn = $("view-report");
const verdictEl = $("verdict");
const errorEl = $("error");

// --- audio state ---
let audioCtx = null;        // AudioContext @ SAMPLE_RATE (capture + playback)
let micStream = null;
let workletNode = null;
let analyser = null;        // taps TTS playback for the lamp
let streaming = false;      // send mic frames only while LISTENING
let playbackChain = Promise.resolve();
let ws = null;
let candidateName = "Candidate";
let pendingReport = null;    // report markdown, shown when "View report" is clicked

// lamp level easing
let targetLevel = 0;
let shownLevel = 0;
let rafId = null;

// Worklet: batch float32 into CHUNK frames, report RMS for the lamp.
const WORKLET_SRC = `
class PCMCollector extends AudioWorkletProcessor {
  constructor() { super(); this._buf = new Float32Array(${CHUNK}); this._n = 0; }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this._buf[this._n++] = ch[i];
      if (this._n === ${CHUNK}) {
        let s = 0;
        for (let k = 0; k < ${CHUNK}; k++) s += this._buf[k] * this._buf[k];
        this.port.postMessage(
          { rms: Math.sqrt(s / ${CHUNK}), pcm: this._buf.slice(0) }
        );
        this._n = 0;
      }
    }
    return true;
  }
}
registerProcessor('pcm-collector', PCMCollector);
`;

// ── lamp + waveform ───────────────────────────────────
function lampLive(on) { onairEl.classList.toggle("live", on); }
function setLabel(text) { onairLabel.textContent = text; }
function setSpeaker(text) { speakerTag.textContent = text; }
// show/animate the waveform whenever a party is talking
function waveActive(on) { waveEl.classList.toggle("active", on); }

function startLampLoop() {
  const tick = () => {
    // ease toward target, normalize a quiet RMS into a lively 0..1
    const norm = Math.min(1, targetLevel * 6);
    shownLevel += (norm - shownLevel) * 0.25;
    // set on the shared parent so lamp and waveform both inherit --lvl
    interviewEl.style.setProperty("--lvl", shownLevel.toFixed(3));
    targetLevel *= 0.85; // decay if no fresh samples arrive
    rafId = requestAnimationFrame(tick);
  };
  if (!rafId) rafId = requestAnimationFrame(tick);
}
function stopLampLoop() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  interviewEl.style.setProperty("--lvl", 0);
  waveActive(false);
}

// ── audio setup ───────────────────────────────────────
async function setupAudio() {
  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  const blobUrl = URL.createObjectURL(
    new Blob([WORKLET_SRC], { type: "application/javascript" })
  );
  await audioCtx.audioWorklet.addModule(blobUrl);

  const source = audioCtx.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(audioCtx, "pcm-collector");
  workletNode.port.onmessage = (e) => {
    if (!streaming) return;
    targetLevel = Math.max(targetLevel, e.data.rms);
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data.pcm.buffer);
  };
  source.connect(workletNode); // worklet is silent; not wired to destination

  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
}

// Decode + play a TTS WAV, feeding the lamp from its amplitude.
function playWav(arrayBuffer) {
  playbackChain = playbackChain.then(
    () =>
      new Promise((resolve) => {
        audioCtx.decodeAudioData(
          arrayBuffer,
          (buffer) => {
            const src = audioCtx.createBufferSource();
            src.buffer = buffer;
            src.connect(analyser);
            analyser.connect(audioCtx.destination);
            const data = new Float32Array(analyser.fftSize);
            const meter = () => {
              if (src.__done) return;
              analyser.getFloatTimeDomainData(data);
              let s = 0;
              for (let i = 0; i < data.length; i++) s += data[i] * data[i];
              targetLevel = Math.max(targetLevel, Math.sqrt(s / data.length));
              requestAnimationFrame(meter);
            };
            src.onended = () => { src.__done = true; resolve(); };
            src.start();
            meter();
          },
          () => resolve()
        );
      })
  );
  return playbackChain;
}

// ── safe-ish markdown ─────────────────────────────────
function renderMarkdown(md) {
  const esc = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc(md)
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/^---$/gm, "<hr>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function showVerdict(md) {
  const m = md.match(/STRONG_HIRE|LEAN_HIRE|NO_HIRE/);
  if (!m) return;
  const v = m[0];
  verdictEl.hidden = false;
  verdictEl.textContent = v.replace("_", " ");
  verdictEl.classList.toggle("no", v === "NO_HIRE");
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.hidden = false;
}

// ── flow ──────────────────────────────────────────────
async function startInterview(e) {
  e.preventDefault();
  errorEl.hidden = true;
  candidateName = $("name").value.trim();
  const file = $("resume").files[0];
  if (!candidateName) { showError("Enter your name to begin."); return; }
  if (!file) { showError("Choose a PDF resume to begin."); return; }

  $("start").disabled = true;
  try {
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch("/upload", { method: "POST", body: form });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.error || `Upload failed (${resp.status}).`);
    }
    const { session_id } = await resp.json();

    await setupAudio();
    startLampLoop();

    setupEl.hidden = true;
    interviewEl.hidden = false;
    setLabel("Connecting");

    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/${session_id}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "start", candidate_name: candidateName }));
      lampLive(true);
      setLabel("On air");
      setSpeaker("Preparing your interview…");
    };

    ws.onmessage = async (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        setLabel("On air");
        setSpeaker("The interviewer is speaking.");
        waveActive(true);          // interviewer talking
        await playWav(ev.data);
        waveActive(false);
        return;
      }
      const msg = JSON.parse(ev.data);
      switch (msg.type) {
        case "status":
          waveActive(false);
          setLabel("On air");
          setSpeaker(msg.message);
          break;
        case "listening":
          await playbackChain;        // don't capture our own TTS
          streaming = true;
          waveActive(true);           // candidate talking
          setLabel("Listening");
          setSpeaker("Your turn — answer out loud.");
          break;
        case "listening_stop":
          streaming = false;
          waveActive(false);
          setLabel("On air");
          setSpeaker("Got it — thinking…");
          break;
        case "transcribed":
          // transcripts are no longer shown live (kept server-side for the report)
          break;
        case "report":
          // the report finished generating while the farewell played; only
          // reveal the button once the farewell has actually finished.
          pendingReport = msg.markdown;
          streaming = false;
          await playbackChain;
          waveActive(false);
          setLabel("Done");
          setSpeaker("That's a wrap — thanks for your time!");
          viewReportBtn.hidden = false;
          break;
        case "error":
          showError(msg.message);
          break;
      }
    };

    ws.onerror = () => showError("Connection error.");
    ws.onclose = () => {
      streaming = false;
      lampLive(false);
      if (reportEl.hidden) { setLabel("Disconnected"); stopLampLoop(); }
    };
  } catch (err) {
    showError(err.message || String(err));
    $("start").disabled = false;
    setupEl.hidden = false;
    interviewEl.hidden = true;
  }
}

function finishInterview(markdown) {
  streaming = false;
  stopLampLoop();
  if (ws) ws.close();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();

  interviewEl.hidden = true;
  reportEl.hidden = false;
  showVerdict(markdown);
  $("report-body").innerHTML = renderMarkdown(markdown);
}

// ── file input label ──────────────────────────────────
const fileInput = $("resume");
const fileHint = $("filename");
fileHint.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  fileHint.textContent = f ? f.name : "No file chosen";
  fileHint.classList.toggle("set", !!f);
});

$("callsheet").addEventListener("submit", startInterview);

// reveal the report (tears down audio/WS) once the candidate clicks
viewReportBtn.addEventListener("click", () => {
  if (pendingReport !== null) finishInterview(pendingReport);
});

// Download the report as PDF via the browser's print-to-PDF.
// The document title seeds the default PDF filename.
$("download").addEventListener("click", () => {
  const prevTitle = document.title;
  document.title = `Interview Report - ${candidateName}`;
  window.print();
  document.title = prevTitle;
});

// Start a fresh interview: reload to reset all UI + audio state cleanly.
$("restart").addEventListener("click", () => location.reload());
