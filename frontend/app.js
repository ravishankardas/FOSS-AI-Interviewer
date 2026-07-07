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
const resumeModal = $("resume-modal");
// coding round
const codingEl = $("coding");
const codingTitleEl = $("coding-title");
const codingPromptEl = $("coding-prompt");
const codingLangEl = $("coding-lang");
const codingOutputEl = $("coding-output");
const codingTestsEl = $("coding-tests");
const codingTimerEl = $("coding-timer");
const runCodeBtn = $("run-code");
const runTestsBtn = $("run-tests");
const submitCodeBtn = $("submit-code");

// --- audio state ---
let audioCtx = null;        // AudioContext @ SAMPLE_RATE (capture + playback)
let micStream = null;
let workletNode = null;
let analyser = null;        // taps TTS playback for the lamp
let streaming = false;      // send mic frames only while LISTENING
let playbackChain = Promise.resolve();
let currentSource = null;   // the AudioBufferSourceNode currently playing (for barge-in)
let bargeAborted = false;   // set when the candidate interrupts; skips queued TTS
let ws = null;
let candidateName = "Candidate";
let pendingReport = null;    // report markdown, shown when "View report" is clicked

// coding state
let editor = null;           // CodeMirror instance (lazy)
let codingActive = false;
const CM_MODE = { python: "python", "c++": "text/x-c++src" };
const STARTER = {
  python:
    "import sys\n\n" +
    "# read input from stdin\n" +
    "data = sys.stdin.read().split()\n\n" +
    "# TODO: write your solution and print the answer\n",
  "c++":
    "#include <iostream>\n#include <string>\n#include <vector>\nusing namespace std;\n\n" +
    "int main() {\n" +
    "    // read input from stdin, e.g.:\n" +
    "    // int x; cin >> x;\n\n" +
    "    // TODO: write your solution and print the answer\n" +
    "    return 0;\n}\n",
};
let codingTimerId = null;

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
// toggles the on-air lamp AND the ambient screen-edge glow together, so the
// whole live session (voice + coding) reads as "the interviewer is watching"
function lampLive(on) {
  onairEl.classList.toggle("live", on);
  document.body.classList.toggle("interview-live", on);
}
function setLabel(text) { onairLabel.textContent = text; }
function setSpeaker(text) { speakerTag.textContent = text; speakerTag.classList.remove("thinking"); }
// like setSpeaker, but appends an animated "…" so the wait (resume parse, LLM,
// TTS synth) reads as active work rather than a frozen line
function setThinking(text) { speakerTag.textContent = text; speakerTag.classList.add("thinking"); }
// show/animate the waveform whenever a party is talking
function waveActive(on) { waveEl.classList.toggle("active", on); }

// live captions — append a line to the interview transcript panel
const transcriptEl = $("transcript");
function addCaption(who, text) {
  if (!text || !text.trim()) return;
  const li = document.createElement("li");
  li.className = "turn " + (who === "you" ? "you" : "interviewer");
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who === "you" ? "You" : "Interviewer";
  const said = document.createElement("span");
  said.className = "said";
  said.textContent = text.trim();
  li.append(label, said);
  transcriptEl.append(li);
  li.scrollIntoView({ block: "nearest" });
}

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
        // candidate barged in: drop this and any further queued clips
        if (bargeAborted) { resolve(); return; }
        audioCtx.decodeAudioData(
          arrayBuffer,
          (buffer) => {
            if (bargeAborted) { resolve(); return; }
            const src = audioCtx.createBufferSource();
            src.buffer = buffer;
            src.connect(analyser);
            analyser.connect(audioCtx.destination);
            currentSource = src;
            const data = new Float32Array(analyser.fftSize);
            const meter = () => {
              if (src.__done) return;
              analyser.getFloatTimeDomainData(data);
              let s = 0;
              for (let i = 0; i < data.length; i++) s += data[i] * data[i];
              targetLevel = Math.max(targetLevel, Math.sqrt(s / data.length));
              requestAnimationFrame(meter);
            };
            src.onended = () => {
              src.__done = true;
              if (currentSource === src) currentSource = null;
              resolve();
            };
            src.start();
            meter();
          },
          () => resolve()
        );
      })
  );
  return playbackChain;
}

// candidate interrupted the interviewer — stop the clip that's playing and let
// the chain drain past anything still queued for this utterance.
function abortPlayback() {
  bargeAborted = true;
  if (currentSource) {
    try { currentSource.stop(); } catch (_) {}
    currentSource = null;
  }
  waveActive(false);
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

let toastTimer = null;
function _toast(msg, { info = false, ms = 4000 } = {}) {
  errorEl.textContent = msg;
  errorEl.classList.toggle("info", info);
  errorEl.hidden = false;
  // re-trigger the slide-in animation even if a toast is already up
  errorEl.style.animation = "none";
  void errorEl.offsetWidth;
  errorEl.style.animation = "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { errorEl.hidden = true; }, ms);
}
function showError(msg) { _toast(msg); }
// non-error nudge — blue toast, lingers a bit longer
function showHint(msg) { _toast(msg, { info: true, ms: 6000 }); }

// count consecutive "not a resume" rejections; after this many in a row we
// stop showing the inline error and pop a modal nudge instead. Reset on a
// successful upload or when a different file is chosen. Purely a UX nudge —
// no server state, so a refresh naturally clears it.
const NOT_RESUME_LIMIT = 3;
let notResumeTries = 0;

function showResumeModal() { resumeModal.hidden = false; }
function hideResumeModal() { resumeModal.hidden = true; }

// ── coding round ──────────────────────────────────────
const KEYWORDS = {
  python: [
    "def", "return", "if", "elif", "else", "for", "while", "in", "range", "len",
    "print", "input", "import", "from", "class", "True", "False", "None", "and",
    "or", "not", "is", "try", "except", "finally", "with", "as", "lambda", "map",
    "filter", "sorted", "reversed", "enumerate", "zip", "int", "str", "list",
    "dict", "set", "tuple", "float", "abs", "min", "max", "sum", "append",
  ],
  "c++": [
    "#include", "int", "long", "double", "float", "char", "bool", "void",
    "return", "if", "else", "for", "while", "do", "switch", "case", "break",
    "continue", "cin", "cout", "endl", "std", "vector", "string", "sort", "swap",
    "push_back", "size", "begin", "end", "auto", "const", "struct", "class",
    "public", "private", "using", "namespace", "true", "false", "nullptr",
  ],
};

// merge language keywords with identifiers already in the buffer
function completer(cm) {
  const cur = cm.getCursor();
  const token = cm.getTokenAt(cur);
  let word = /[\w#]+/.test(token.string) ? token.string : "";
  const start = word ? token.start : cur.ch;
  const lang = codingLangEl.value || "python";
  const fromBuffer = cm.getValue().match(/[A-Za-z_]\w{2,}/g) || [];
  const pool = new Set([...(KEYWORDS[lang] || []), ...fromBuffer]);
  const list = [...pool]
    .filter((w) => w.toLowerCase().startsWith(word.toLowerCase()) && w !== word)
    .sort()
    .slice(0, 30);
  return {
    list,
    from: CodeMirror.Pos(cur.line, start),
    to: CodeMirror.Pos(cur.line, cur.ch),
  };
}

function ensureEditor() {
  if (editor) return editor;
  editor = CodeMirror.fromTextArea($("code-editor"), {
    mode: CM_MODE.python,
    theme: "default",
    lineNumbers: true,
    indentUnit: 4,
    tabSize: 4,
    smartIndent: true,       // mode-aware auto-indent (e.g. after `:` in Python)
    electricChars: true,
    autoCloseBrackets: true, // auto-insert closing ) ] } " '
    matchBrackets: true,
    autofocus: true,
    extraKeys: {
      "Ctrl-Space": (cm) => cm.showHint({ hint: completer, completeSingle: false }),
      Tab: (cm) => {
        if (cm.somethingSelected()) cm.indentSelection("add");
        else cm.replaceSelection(" ".repeat(cm.getOption("indentUnit")), "end");
      },
      Enter: "newlineAndIndent",
    },
  });
  // editing invalidates the last passing test run — re-lock Submit
  editor.on("change", () => {
    if (allTestsPassed) {
      allTestsPassed = false;
      refreshSubmitBtn();
    }
    scheduleCodeState();   // keep the interviewer's view of the code current
  });
  // pop completions as you type a word (not on every keystroke)
  editor.on("inputRead", (cm, change) => {
    if (cm.state.completionActive) return;
    const ch = change.text[0];
    if (ch && /[A-Za-z_]/.test(ch)) {
      cm.showHint({ hint: completer, completeSingle: false });
    }
  });
  return editor;
}

let currentTests = [];   // visible test cases for the active coding question
let currentStarter = {}; // per-language starter code for the active question
let codingBusy = false;  // an execution request is in flight
let allTestsPassed = false; // every visible test passed on the last run

// the starter for a language: problem-specific if provided, else generic fallback
function starterFor(lang) {
  return (currentStarter && currentStarter[lang]) || STARTER[lang];
}

function setCodingBusy(busy) {
  codingBusy = busy;
  runCodeBtn.disabled = busy;
  runTestsBtn.disabled = busy;
  refreshSubmitBtn();
}

// Submit stays locked until every visible test passes (and no run is in flight).
// Editing the code after a green run re-locks it — the passing run no longer
// reflects what's in the editor.
function refreshSubmitBtn() {
  submitCodeBtn.disabled = codingBusy || !allTestsPassed;
  submitCodeBtn.title = allTestsPassed
    ? ""
    : "Pass all visible test cases to unlock Submit";
}

// ── coding timer ──────────────────────────────────────
function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function startTimer(seconds) {
  stopTimer();
  let left = seconds;
  const tick = () => {
    codingTimerEl.textContent = fmtTime(left);
    codingTimerEl.classList.toggle("low", left <= 60 && left > 20);
    codingTimerEl.classList.toggle("critical", left <= 20);
    if (left <= 0) {
      stopTimer();
      // time's up — auto-submit whatever is in the editor
      submitCoding(true);
      return;
    }
    left -= 1;
  };
  tick();
  codingTimerId = setInterval(tick, 1000);
}

function stopTimer() {
  if (codingTimerId) clearInterval(codingTimerId);
  codingTimerId = null;
}

function submitCoding(timedOut = false) {
  if (!codingActive) return;
  stopTimer();
  setCodingBusy(true);
  sendCode("code_submit");
  hideCoding();
  setLabel("On air");
  setSpeaker(timedOut ? "Time's up — reviewing your solution…" : "Reviewing your solution…");
}

const esc = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// render the test list; pass `results` to show pass/fail + actual output
function renderTests(results) {
  if (!currentTests.length) {
    codingTestsEl.innerHTML = "";
    return;
  }
  codingTestsEl.innerHTML = currentTests
    .map((t, i) => {
      const r = results && results[i];
      const cls = r ? (r.passed ? "pass" : "fail") : "";
      const status = r ? (r.passed ? "pass" : "fail") : "not run";
      const stdinRow = t.stdin
        ? `<div><span class="k">in:</span>${esc(t.stdin.replace(/\n+$/, "")).replace(/\n/g, "<br>")}</div>`
        : "";
      const expRow = `<div><span class="k">want:</span>${esc(t.expected)}</div>`;
      const gotRow =
        r && !r.passed
          ? `<div><span class="k">got:</span><span class="got-bad">${esc(
              r.error || r.actual || "(no output)"
            )}</span></div>`
          : "";
      return `<li class="${cls}">
        <div class="test-head"><span class="test-status">${status}</span><span>${esc(
        t.name
      )}</span></div>
        <div class="test-io">${stdinRow}${expRow}${gotRow}</div>
      </li>`;
    })
    .join("");
}

// show the editor for a coding_question; the spoken intro arrives as a separate
// binary frame and plays via the normal playWav path.
function showCoding(msg) {
  codingActive = true;
  codingTitleEl.textContent = msg.title || "Coding problem";
  codingPromptEl.textContent = msg.prompt || "";
  codingOutputEl.textContent = "Run your code, or run the test cases below.";
  codingOutputEl.className = "coding-output";
  currentTests = msg.tests || [];
  currentStarter = msg.starter || {};
  allTestsPassed = false;   // must pass the visible tests before submitting
  renderTests(null);

  interviewEl.hidden = true;
  codingEl.hidden = false;
  setLabel("Coding");

  const cm = ensureEditor();
  const lang = codingLangEl.value || "python";
  cm.setOption("mode", CM_MODE[lang]);
  cm.setValue(starterFor(lang));
  setCodingBusy(false);
  startTimer(msg.time_limit || 300);
  // the interviewer listens throughout the coding round — stream mic continuously
  // so spoken questions / hint requests are heard (server VAD gates out its own TTS)
  streaming = true;
  sendCode("code_state");   // give the interviewer the starting screen state
  // one-time nudge: many candidates don't realise the round is conversational
  showHint("You can talk to the interviewer during the coding round,  ask a question or for a hint any time.");
  // CodeMirror needs a refresh once its container is visible
  setTimeout(() => cm.refresh(), 0);
}

function hideCoding() {
  codingActive = false;
  streaming = false;        // stop listening; the verbal round re-enables per turn
  stopTimer();
  codingEl.hidden = true;
  interviewEl.hidden = false;
}

// re-open the editor for an optimization pass — keep the candidate's code, but
// require them to re-verify tests before submitting the improved version
function reopenCoding(timeLimit) {
  codingActive = true;
  interviewEl.hidden = true;
  codingEl.hidden = false;
  setLabel("Coding");
  streaming = true;
  allTestsPassed = false;
  refreshSubmitBtn();
  setCodingBusy(false);
  codingOutputEl.textContent = "Refine your solution, re-run the tests, and submit again.";
  codingOutputEl.className = "coding-output";
  startTimer(timeLimit || 300);
  setTimeout(() => editor && editor.refresh(), 0);
}

function renderRunResult(msg) {
  setCodingBusy(false);
  if (msg.error) {
    codingOutputEl.textContent = msg.error;
    codingOutputEl.className = "coding-output err";
    return;
  }
  if (msg.compile_error) {
    codingOutputEl.textContent = msg.compile_error;
    codingOutputEl.className = "coding-output err";
    return;
  }
  let text = msg.stdout || "";
  if (msg.stderr) text += (text ? "\n" : "") + msg.stderr;
  if (msg.timed_out) text += (text ? "\n" : "") + "[timed out]";
  if (!text) text = `(no output, exit code ${msg.exit_code})`;
  codingOutputEl.textContent = text;
  codingOutputEl.className =
    "coding-output " + (msg.exit_code === 0 && !msg.stderr ? "ok" : "");
}

function sendCode(type) {
  if (!ws || ws.readyState !== WebSocket.OPEN || !editor) return;
  ws.send(
    JSON.stringify({
      type,
      language: codingLangEl.value || "python",
      code: editor.getValue(),
    })
  );
}

// push the live editor contents to the interviewer, debounced so we send on a
// typing pause rather than every keystroke
let codeStateTimer = null;
function scheduleCodeState() {
  if (!codingActive) return;
  clearTimeout(codeStateTimer);
  codeStateTimer = setTimeout(() => sendCode("code_state"), 700);
}

// ── flow ──────────────────────────────────────────────
async function startInterview(e) {
  e.preventDefault();
  errorEl.hidden = true;
  candidateName = $("name").value.trim();
  const file = $("resume").files[0];
  if (!candidateName) { showError("Enter your name to begin."); return; }
  if (!file) { showError("Choose a PDF resume to begin."); return; }

  const startBtn = $("start");
  const startHTML = startBtn.innerHTML;
  const restoreStart = () => {
    startBtn.disabled = false;
    startBtn.classList.remove("loading");
    startBtn.innerHTML = startHTML;
  };
  startBtn.disabled = true;
  startBtn.classList.add("loading");
  startBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span>Checking résumé…';
  try {
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch("/upload", { method: "POST", body: form });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      if (body.reason === "not_resume") {
        notResumeTries += 1;
        if (notResumeTries >= NOT_RESUME_LIMIT) {
          notResumeTries = 0;
          showResumeModal();
        } else {
          showError(body.error);
        }
        restoreStart();
        return;
      }
      throw new Error(body.error || `Upload failed (${resp.status}).`);
    }
    notResumeTries = 0;
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
      setThinking("Preparing your interview");
    };

    ws.onmessage = async (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        // a fresh interviewer utterance (we're between turns, not streaming) —
        // clear any leftover barge-abort so this clip actually plays
        if (!streaming) bargeAborted = false;
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
          if (msg.barge) {
            // barge-in enabled: start streaming mic frames NOW, while the
            // interviewer is still talking, so the server can hear an interruption.
            // Echo cancellation keeps our own TTS out of the frames.
            streaming = true;
            setLabel("Listening");
            // keep the "interviewer is speaking" cue until playback finishes on
            // its own; if it does (no interruption) prompt for the answer
            playbackChain.then(() => {
              if (streaming && !bargeAborted) {
                waveActive(true);
                setSpeaker("Your turn — answer out loud.");
              }
            });
          } else {
            await playbackChain;      // don't capture our own TTS
            streaming = true;
            waveActive(true);         // candidate talking
            setLabel("Listening");
            setSpeaker("Your turn — answer out loud.");
          }
          break;
        case "barge_in":
          // the server's VAD heard us start talking over the interviewer — cut it
          abortPlayback();
          waveActive(true);           // candidate talking now
          setLabel("Listening");
          setSpeaker("Go ahead — I'm listening.");
          break;
        case "listening_stop":
          streaming = false;
          waveActive(false);
          setLabel("On air");
          setThinking("Got it — thinking");
          break;
        case "caption":
          // live captions: the interviewer just said something
          addCaption(msg.who || "interviewer", msg.text);
          break;
        case "transcribed":
          // the candidate's answer, transcribed — show it as a caption line
          addCaption("you", msg.text);
          break;
        case "coding_question":
          showCoding(msg);
          break;
        case "optimize_prompt":
          // interviewer asked to improve a working-but-suboptimal solution
          reopenCoding(msg.time_limit);
          break;
        case "run_result":
          renderRunResult(msg);
          break;
        case "test_results": {
          renderTests(msg.results);
          const passed = msg.results.filter((r) => r.passed).length;
          allTestsPassed =
            msg.results.length > 0 && passed === msg.results.length;
          setCodingBusy(false);
          codingOutputEl.textContent = allTestsPassed
            ? `${passed}/${msg.results.length} test cases passed — you can submit now.`
            : `${passed}/${msg.results.length} test cases passed.`;
          codingOutputEl.className =
            "coding-output " + (allTestsPassed ? "ok" : "err");
          break;
        }
        case "engine_error":
          // execution engine down: tell the candidate it's infra, not their
          // code, and clear any stale pass/fail so nothing reads as a failure.
          setCodingBusy(false);
          renderTests(null);
          codingOutputEl.textContent = msg.message;
          codingOutputEl.className = "coding-output err";
          break;
        case "reload":
          // interview ended early (e.g. silence). Let the goodbye finish playing,
          // then reset the whole app by reloading.
          await playbackChain;
          location.reload();
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
    restoreStart();
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
  notResumeTries = 0;   // a different file — start the count fresh
});

$("resume-modal-ok").addEventListener("click", hideResumeModal);

$("callsheet").addEventListener("submit", startInterview);

// coding round controls
runCodeBtn.addEventListener("click", () => {
  if (!codingActive) return;
  codingOutputEl.textContent = "Running…";
  codingOutputEl.className = "coding-output";
  setCodingBusy(true);
  sendCode("run_code");
});

runTestsBtn.addEventListener("click", () => {
  if (!codingActive) return;
  codingOutputEl.textContent = "Running test cases…";
  codingOutputEl.className = "coding-output";
  setCodingBusy(true);
  sendCode("run_tests");
});

submitCodeBtn.addEventListener("click", () => submitCoding(false));

// switch editor language; swap in the starter only if untouched
codingLangEl.addEventListener("change", () => {
  if (!editor) return;
  const lang = codingLangEl.value;
  editor.setOption("mode", CM_MODE[lang]);
  const cur = editor.getValue().trim();
  const untouched = [
    "",
    starterFor("python").trim(),
    starterFor("c++").trim(),
    STARTER.python.trim(),
    STARTER["c++"].trim(),
  ].includes(cur);
  if (untouched) editor.setValue(starterFor(lang));
});

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

// Mid-interview restart (interview + coding stages): confirm first, since this
// abandons the in-progress session, then reload for the same clean reset.
["restart-interview", "restart-coding"].forEach((id) => {
  const el = $(id);
  if (el) {
    el.addEventListener("click", () => {
      if (confirm("Restart the interview? Your current progress will be lost.")) {
        location.reload();
      }
    });
  }
});

// ── interview history ─────────────────────────────────
const historyEl = $("history");
const historyListEl = $("history-list");
const historyDetailEl = $("history-detail");
const historyEmptyEl = $("history-empty");
const historyDownloadEl = $("history-download");
let historyName = "";   // candidate name of the currently open history detail

function fmtDate(iso) {
  const d = new Date(iso);
  return isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

async function openHistory() {
  errorEl.hidden = true;
  setupEl.hidden = true;
  historyEl.hidden = false;
  historyDetailEl.hidden = true;
  historyListEl.hidden = false;
  historyEmptyEl.hidden = true;
  historyDownloadEl.hidden = true;   // only shown when a detail is open
  historyListEl.innerHTML = "";
  try {
    const resp = await fetch("/history");
    const { interviews } = await resp.json();
    if (!interviews || !interviews.length) { historyEmptyEl.hidden = false; return; }
    for (const it of interviews) historyListEl.append(historyRow(it));
  } catch {
    showError("Couldn't load your interview history.");
  }
}

function historyRow(it) {
  const li = document.createElement("li");
  const rec = (it.recommendation || "").toUpperCase();
  const recClass = rec === "NO_HIRE" ? "no" : (rec.includes("HIRE") ? "hire" : "");

  const left = document.createElement("div");
  const name = document.createElement("div");
  name.className = "h-name";
  name.textContent = it.candidate_name || "Candidate";
  const date = document.createElement("div");
  date.className = "h-date";
  date.textContent = fmtDate(it.created_at);
  left.append(name, date);

  const right = document.createElement("div");
  right.style.textAlign = "right";
  const score = document.createElement("div");
  score.className = "h-score";
  score.textContent = (it.avg_score != null ? it.avg_score : "–") + " / 10";
  const recTag = document.createElement("div");
  recTag.className = "h-rec " + recClass;
  recTag.textContent = rec.replace("_", " ") || "—";
  right.append(score, recTag);

  li.append(left, right);
  li.addEventListener("click", () => openHistoryDetail(it.id));
  return li;
}

async function openHistoryDetail(id) {
  try {
    const resp = await fetch(`/history/${id}`);
    if (!resp.ok) throw new Error();
    const rec = await resp.json();
    historyName = rec.candidate_name || "Candidate";
    historyDetailEl.innerHTML = renderMarkdown(rec.markdown || "");
    historyListEl.hidden = true;
    historyEmptyEl.hidden = true;
    historyDetailEl.hidden = false;
    historyDownloadEl.hidden = false;
  } catch {
    showError("Couldn't load that interview.");
  }
}

// Download a past interview's report as PDF (same print-to-PDF path as the
// end-of-interview download). The title seeds the default filename.
historyDownloadEl.addEventListener("click", () => {
  const prevTitle = document.title;
  document.title = `Interview Report - ${historyName}`;
  window.print();
  document.title = prevTitle;
});

$("show-history").addEventListener("click", openHistory);
$("history-back").addEventListener("click", () => {
  if (!historyDetailEl.hidden) {
    openHistory();            // detail open → back to the list
  } else {
    historyEl.hidden = true;  // list open → back to setup
    setupEl.hidden = false;
  }
});

// ── dev preview mode ──────────────────────────────────
// Jump straight to a stage with mock data — no upload, no WebSocket — so you can
// tweak the UI and just hard-refresh instead of re-running the whole interview.
//   ?preview=coding     the coding editor with a sample problem
//   ?preview=interview  the live "on air" voice stage
//   ?preview=report     a sample report
// Has no effect on the normal flow (only runs when ?preview=... is present).
(function initPreview() {
  // dev-only: never runs off localhost, so it's inert in any real deployment
  const local = ["localhost", "127.0.0.1", "::1", ""].includes(location.hostname);
  const stage = new URLSearchParams(location.search).get("preview");
  if (!stage || !local) return;
  setupEl.hidden = true;

  if (stage === "coding") {
    showCoding({
      title: "Two Sum (preview)",
      prompt: "Read a line of space-separated integers, then a target on the next "
        + "line. Print the 0-based indices of the two numbers that add up to the "
        + "target.\n\nInput:\n  2 7 11 15\n  9\nOutput:\n  0 1",
      time_limit: 300,
      starter: {},
      tests: [
        { name: "example", stdin: "2 7 11 15\n9\n", expected: "0 1" },
        { name: "adjacent", stdin: "3 3\n6\n", expected: "0 1" },
      ],
    });
    lampLive(true);
  } else if (stage === "interview") {
    interviewEl.hidden = false;
    startLampLoop();
    lampLive(true);
    setLabel("On air");
    setSpeaker("Preview — the interviewer is speaking.");
    waveActive(true);
  } else if (stage === "history") {
    openHistory();
  } else if (stage === "report") {
    reportEl.hidden = false;
    $("report-body").innerHTML = renderMarkdown(
      "# Interview Report - Preview\n\n## Per-Question Breakdown\n\n"
      + "### Q1: Tell me about a project (Topic: experience)\n"
      + "- **Score:** 8\n- **Feedback:** Strong, concrete example.\n\n"
      + "## Overall Summary\nSolid across the board.\n\n"
      + "## Recommendation\nLEAN_HIRE\n"
    );
  }
})();
