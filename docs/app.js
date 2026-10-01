"use strict";
(() => {
  const data = window.BSWS_DATA;
  const status = document.getElementById("audio-status");
  if (!data) { status.textContent = "The study data could not be loaded. Please reload this page or use the downloads below."; return; }
  const select = document.getElementById("utterance");
  const players = document.getElementById("players");
  const audios = [];
  const labels = Object.fromEntries(data.models.map(model => [model.key, model.label]));
  // Remove floating-point summation noise only when rounding for display.
  const f = (n, digits = 3) => {
    const rounded = Math.round((Math.abs(Number(n)) + 1e-12) * 10 ** digits) / 10 ** digits;
    return (Number(n) < 0 && rounded !== 0 ? "−" : "") + rounded.toFixed(digits);
  };
  const signed = (n, digits = 3) => (n > 0 ? "+" : "") + f(n, digits);
  const ci = values => `[${f(values[0])}, ${f(values[1])}]`;
  const cell = (row, text, header = false) => { const el = document.createElement(header ? "th" : "td"); el.textContent = text; if(header) el.scope = "row"; row.append(el); return el; };
  for (const utterance of data.utterances) { const option = document.createElement("option"); option.value = utterance.id; option.textContent = utterance.id; select.append(option); }
  for (const model of data.models) {
    const panel = document.createElement("article"); panel.className = "player";
    const title = document.createElement("h3"); title.textContent = model.label; panel.append(title);
    const kind = document.createElement("p"); kind.className = "player-kind"; kind.textContent = model.role; panel.append(kind);
    const audio = document.createElement("audio"); audio.controls = true; audio.preload = "none"; audio.dataset.model = model.key;
    audio.addEventListener("play", () => { for (const other of audios) if (other !== audio) { other.pause(); other.currentTime = 0; } status.textContent = ""; });
    audio.addEventListener("error", () => { status.textContent = `Could not load ${select.value}, ${model.label}. Try the WAV download or reload the page.`; });
    panel.append(audio); audios.push(audio);
    const download = document.createElement("a"); download.className = "audio-download"; download.textContent = "Download WAV"; download.download = ""; panel.append(download);
    players.append(panel);
  }
  const changeUtterance = () => {
    const utterance = data.utterances.find(u => u.id === select.value);
    for (const audio of audios) {
      audio.pause();
      const asset = utterance.audio[audio.dataset.model];
      audio.src = asset.path;
      audio.setAttribute("aria-label", `${utterance.id} — ${labels[audio.dataset.model]}`);
      const download = audio.nextElementSibling; download.href = asset.path; download.download = `${utterance.id}_${audio.dataset.model}.wav`;
      download.setAttribute("aria-label", `Download ${utterance.id}, ${labels[audio.dataset.model]}, WAV`);
      audio.load();
    }
    document.getElementById("utterance-info").textContent = `${utterance.reference_id} · GT ${f(utterance.duration, 2)} s`;
    document.getElementById("previous").disabled = select.selectedIndex === 0;
    document.getElementById("next").disabled = select.selectedIndex === data.utterances.length - 1;
    status.textContent = "";
  };
  select.addEventListener("change", changeUtterance);
  document.getElementById("previous").addEventListener("click", () => { if(select.selectedIndex > 0) { select.selectedIndex--; changeUtterance(); } });
  document.getElementById("next").addEventListener("click", () => { if(select.selectedIndex < data.utterances.length - 1) { select.selectedIndex++; changeUtterance(); } });
  changeUtterance();

  for (const key of data.results.model_order) {
    const result = data.results.mos[key]; const row = document.createElement("tr");
    if (["Snake_MPD_MRD", "LeakyReLU_MPD_MRD"].includes(key)) row.className = "group-start";
    cell(row, labels[key], true); cell(row, f(result.mean)); cell(row, ci(result.ci95)); document.getElementById("mos-rows").append(row);
  }
  for (const activation of ["Snake", "LeakyReLU"]) {
    for (const [container, field] of [["cmos-rows", "cmos_full_minus_mrd_only"], ["paired-rows", "mos_paired_full_minus_mrd_only"]]) {
      const result = data.results[field][activation]; const row = document.createElement("tr");
      cell(row, activation, true); cell(row, signed(result.mean)); cell(row, ci(result.ci95)); document.getElementById(container).append(row);
    }
  }
  const names = {mcd:"MCD ↓",plcc:"PLCC ↑",ssim:"SSIM ↑",mstft:"M-STFT ↓",pesq:"PESQ ↑",periodicity_error:"Periodicity error ↓",pitch_error:"Pitch error ↓",voicing_f1:"Voicing F1 ↑",utmos:"UTMOS ↑"};
  for (const activation of ["Snake", "LeakyReLU"]) {
    const group = document.createElement("div"); group.className = "objective-table";
    const title = document.createElement("h3"); title.textContent = `${activation} · matched listening stimuli`; group.append(title);
    const scroll = document.createElement("div"); scroll.className = "table-scroll"; scroll.tabIndex = 0; scroll.setAttribute("role", "region"); scroll.setAttribute("aria-label", `${activation} objective results`);
    const table = document.createElement("table"); const caption = document.createElement("caption"); caption.className = "sr-only"; caption.textContent = `${activation}: objective means and direction-adjusted differences with paired-utterance bootstrap intervals`; table.append(caption);
    const head = table.createTHead(); const hr = head.insertRow();
    for (const text of ["Metric", "MPD+MRD", "MRD-only", "Adjusted Δ", "95% CI for Δ"]) { const th = document.createElement("th"); th.scope = "col"; th.textContent = text; hr.append(th); }
    const body = table.createTBody();
    for (const result of data.objective.filter(x => x.activation === activation)) {
      const row = body.insertRow(); if(result.metric === "utmos") row.className = "group-start";
      cell(row, names[result.metric], true); cell(row, f(result.mean_full,4)); cell(row, f(result.mean_mrd_only,4)); cell(row, signed(result.benefit_full,4)); cell(row, `[${f(result.ci95_low,4)}, ${f(result.ci95_high,4)}]`);
    }
    scroll.append(table); group.append(scroll); document.getElementById("objective-tables").append(group);
  }
})();
