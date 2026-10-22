/* Network Intrusion Detection - frontend logic (plain JavaScript, no libraries) */
let META = null;

const $ = (id) => document.getElementById(id);
const pct = (x, d = 2) => (x * 100).toFixed(d) + "%";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const show = (el, on = true) => el.classList.toggle("hidden", !on);

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
  });
});

/* ---------- overview ---------- */
function renderKpis(m) {
  const d = m.class_distribution;
  const total = d.normal + d.anomaly;
  const items = [
    [m.n_records.toLocaleString(), "Connection records"],
    [m.n_features, "Input features"],
    [pct(d.normal / total, 1), "Normal traffic"],
    [pct(d.anomaly / total, 1), "Anomaly traffic"],
    [m.n_test.toLocaleString(), "Test records (unseen)"],
  ];
  $("kpis").innerHTML = items.map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
}

function renderMetrics(m) {
  const keys = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1-score"], ["roc_auc", "ROC-AUC"]];
  const rf = m.metrics.rf, dn = m.metrics.dnn;
  const rows = keys.map(([k, name]) => {
    const rfBest = rf[k] >= dn[k], dnBest = dn[k] > rf[k];
    const fmt = (v) => (k === "roc_auc" ? v.toFixed(4) : pct(v));
    return `<tr><td>${name}</td>
      <td class="${rfBest ? "best" : ""}">${fmt(rf[k])}</td>
      <td class="${dnBest ? "best" : ""}">${fmt(dn[k])}</td>
      <td style="width:38%"><div class="bar green"><i style="width:${rf[k] * 100}%"></i></div>
          <div class="bar navy" style="margin-top:3px"><i style="width:${dn[k] * 100}%"></i></div></td></tr>`;
  }).join("");
  $("metricsTable").innerHTML = `<table class="table">
    <tr><th>Metric</th><th>Random Forest</th><th>Deep Neural Network</th><th>Visual (green = RF, navy = DNN)</th></tr>${rows}</table>`;
}

function renderCm(el, cm) {
  const [[tn, fp], [fn, tp]] = cm;
  $(el).innerHTML = `<div class="cm">
    <div></div><div class="h">Predicted<br>normal</div><div class="h">Predicted<br>anomaly</div>
    <div class="h">Actual<br>normal</div>
      <div class="c ok">${tn.toLocaleString()}<small>true negative</small></div>
      <div class="c bad">${fp.toLocaleString()}<small>false alarm</small></div>
    <div class="h">Actual<br>anomaly</div>
      <div class="c bad">${fn.toLocaleString()}<small>missed attack</small></div>
      <div class="c ok">${tp.toLocaleString()}<small>true positive</small></div>
  </div>`;
}

function renderRoc(roc, metrics) {
  const S = 280, P = 36;
  const pts = (r) => r.fpr.map((x, i) => `${P + x * S},${P + S - r.tpr[i] * S}`).join(" ");
  $("roc").innerHTML = `<svg viewBox="0 0 ${S + P * 2} ${S + P * 2}" style="width:100%;max-width:380px">
    <rect x="${P}" y="${P}" width="${S}" height="${S}" fill="#fff" stroke="#dce4ec"/>
    <line x1="${P}" y1="${P + S}" x2="${P + S}" y2="${P}" stroke="#9aa9b9" stroke-dasharray="4 4"/>
    <polyline points="${pts(roc.rf)}" fill="none" stroke="#12a37f" stroke-width="2.5"/>
    <polyline points="${pts(roc.dnn)}" fill="none" stroke="#16324f" stroke-width="2.5"/>
    <text x="${P + S / 2}" y="${P + S + 28}" text-anchor="middle" font-size="11" fill="#6b7c8f">False positive rate</text>
    <text x="12" y="${P + S / 2}" font-size="11" fill="#6b7c8f" transform="rotate(-90 12 ${P + S / 2})" text-anchor="middle">True positive rate</text>
    <text x="${P + S - 8}" y="${P + S - 30}" text-anchor="end" font-size="12" fill="#12a37f">Random Forest (AUC ${metrics.rf.roc_auc.toFixed(4)})</text>
    <text x="${P + S - 8}" y="${P + S - 12}" text-anchor="end" font-size="12" fill="#16324f">DNN (AUC ${metrics.dnn.roc_auc.toFixed(4)})</text>
  </svg>`;
}

function renderImportance(list) {
  const max = list[0].importance;
  $("importance").innerHTML = list.map((it) =>
    `<div class="bar-row"><span title="${esc(it.feature)}">${esc(it.feature)}</span>
       <div class="bar"><i style="width:${(it.importance / max) * 100}%"></i></div>
       <span>${(it.importance * 100).toFixed(1)}%</span></div>`).join("");
}

/* ---------- live detection form ---------- */
function buildForm(meta) {
  const html = meta.groups.map((g) => {
    const feats = meta.features.filter((f) => f.group === g);
    if (!feats.length) return "";
    const fields = feats.map((f) => {
      if (f.kind === "categorical") {
        const opts = f.categories.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
        return `<div class="field"><label>${esc(f.name)}</label><select data-feature="${esc(f.name)}">${opts}</select></div>`;
      }
      return `<div class="field"><label>${esc(f.name)}</label>
        <input type="number" step="any" data-feature="${esc(f.name)}" value="${f.median}" title="Range in training data: ${f.min} to ${f.max}"></div>`;
    }).join("");
    return `<fieldset><legend>${esc(g)}</legend><div class="fields">${fields}</div></fieldset>`;
  }).join("");
  $("formGroups").innerHTML = html;
}

function fillForm(features) {
  document.querySelectorAll("[data-feature]").forEach((el) => {
    const v = features[el.dataset.feature];
    if (v !== undefined) el.value = v;
  });
}

async function loadSample(label) {
  const res = await fetch("/api/sample?label=" + label);
  const data = await res.json();
  fillForm(data.features);
  const note = $("sampleNote");
  note.textContent = `Loaded a real ${data.actual.toUpperCase()} connection from the test data. Click "Analyse connection" to see what the models say.`;
  show(note);
  show($("result"), false);
}

function renderResult(r) {
  const flagged = r.rf.label === "anomaly" || r.dnn.label === "anomaly";
  const both = r.rf.label === "anomaly" && r.dnn.label === "anomaly";
  const title = both ? "Intrusion detected" : flagged ? "Suspicious connection" : "Normal traffic";
  const msg = both ? "Both models classify this connection as an anomaly."
    : flagged ? "The models disagree - one flags this connection as an anomaly. Worth investigating."
    : "Both models classify this connection as normal.";
  const row = (name, m, cls) => `
    <div class="bar-row" style="grid-template-columns:150px 1fr 150px">
      <b>${name}</b>
      <div class="bar ${m.label === "anomaly" ? "red" : "green"}"><i style="width:${m.probability * 100}%"></i></div>
      <span><span class="badge ${m.label === "anomaly" ? "bad" : "good"}">${m.label.toUpperCase()}</span> ${pct(m.probability, 1)}</span>
    </div>`;
  $("result").innerHTML = `
    <div class="verdict ${flagged ? "bad" : "good"}"><h3>${title}</h3><p>${msg}</p></div>
    <div class="card"><h2>Intrusion probability</h2>
      ${row("Random Forest", r.rf)}${row("Deep Neural Network", r.dnn)}
      <p class="muted">Probability that the connection is an intrusion. 50% or more is flagged as an anomaly.</p></div>`;
  show($("result"));
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyse() {
  const err = $("formError");
  show(err, false);
  const features = {};
  let bad = null;
  document.querySelectorAll("[data-feature]").forEach((el) => {
    if (el.value === "") bad = el.dataset.feature;
    features[el.dataset.feature] = el.tagName === "SELECT" ? el.value : Number(el.value);
  });
  if (bad) { err.textContent = `Please fill in "${bad}".`; show(err); return; }
  $("btnAnalyze").disabled = true;
  try {
    const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResult(data);
  } catch (e) { err.textContent = e.message; show(err); }
  $("btnAnalyze").disabled = false;
}

/* ---------- batch ---------- */
function batchForm() {
  const f = $("csvFile").files[0];
  if (!f) return null;
  const fd = new FormData();
  fd.append("file", f);
  return fd;
}

async function analyseBatch() {
  const err = $("batchError");
  show(err, false);
  const fd = batchForm();
  if (!fd) { err.textContent = "Please choose a CSV file first."; show(err); return; }
  $("btnBatch").disabled = true;
  try {
    const res = await fetch("/api/predict_csv", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderBatch(data);
  } catch (e) { err.textContent = e.message; show(err); show($("batchResult"), false); }
  $("btnBatch").disabled = false;
}

function renderBatch(data) {
  const s = data.summary;
  const kpi = (v, l) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  let kpis = kpi(s.rows.toLocaleString(), "Rows analysed") + kpi(s.rf_anomalies.toLocaleString(), "Flagged by Random Forest")
    + kpi(s.dnn_anomalies.toLocaleString(), "Flagged by DNN") + kpi(pct(s.models_agree / s.rows, 1), "Models agree");
  if (s.rf_accuracy !== undefined) kpis += kpi(pct(s.rf_accuracy, 1), "RF accuracy vs class") + kpi(pct(s.dnn_accuracy, 1), "DNN accuracy vs class");
  const badge = (l) => `<span class="badge ${l === "anomaly" ? "bad" : "good"}">${l}</span>`;
  const rows = data.rows.map((r) => `<tr><td>${r.row}</td>
      <td>${badge(r.rf_label)} ${pct(r.rf_probability, 1)}</td><td>${badge(r.dnn_label)} ${pct(r.dnn_probability, 1)}</td>
      <td>${r.actual ? badge(r.actual) : "-"}</td></tr>`).join("");
  $("batchResult").innerHTML = `
    <div class="kpis">${kpis}</div>
    ${s.truncated ? `<div class="note">Only the first 20,000 rows were analysed.</div>` : ""}
    <div class="card"><h2>Predictions <span class="muted">${data.rows.length < s.rows ? `(first ${data.rows.length} of ${s.rows.toLocaleString()} rows shown - download the CSV for all)` : ""}</span></h2>
      <div class="scroll"><table class="table"><tr><th>Row</th><th>Random Forest</th><th>Deep Neural Network</th><th>Actual class</th></tr>${rows}</table></div></div>`;
  show($("batchResult"));
}

async function downloadBatch() {
  const err = $("batchError");
  show(err, false);
  const fd = batchForm();
  if (!fd) { err.textContent = "Please choose a CSV file first."; show(err); return; }
  const res = await fetch("/api/predict_csv?download=1", { method: "POST", body: fd });
  if (!res.ok) { const d = await res.json(); err.textContent = d.error || "Request failed"; show(err); return; }
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url; a.download = "nids_predictions.csv"; a.click();
  URL.revokeObjectURL(url);
}

/* ---------- start ---------- */
async function init() {
  const res = await fetch("/api/summary");
  META = await res.json();
  renderKpis(META);
  renderMetrics(META);
  renderCm("cmRf", META.confusion.rf);
  renderCm("cmDnn", META.confusion.dnn);
  renderRoc(META.roc, META.metrics);
  renderImportance(META.importances);
  buildForm(META);
  $("backendNote").textContent = META.dnn_backend === "keras"
    ? "(Built with TensorFlow/Keras.)" : "(Built with scikit-learn MLPClassifier, same architecture - TensorFlow was not installed.)";
  loadSample("any");

  $("btnNormal").onclick = () => loadSample("normal");
  $("btnAttack").onclick = () => loadSample("anomaly");
  $("btnAny").onclick = () => loadSample("any");
  $("btnAnalyze").onclick = analyse;
  $("btnBatch").onclick = analyseBatch;
  $("btnDownload").onclick = downloadBatch;
}
init();
