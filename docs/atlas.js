"use strict";
(() => {
  const data = window.BSWS_DATA;
  const select = document.getElementById("atlas-utterance");
  const image = document.getElementById("atlas-image");
  const viewport = document.getElementById("atlas-viewport");
  const status = document.getElementById("atlas-status");
  const retry = document.getElementById("atlas-retry");
  if (!data || !select || !image) return;

  for (const utterance of data.utterances) {
    const option = document.createElement("option");
    option.value = utterance.id;
    option.textContent = utterance.id;
    select.append(option);
  }
  const previous = document.getElementById("atlas-previous");
  const next = document.getElementById("atlas-next");
  let request = 0;
  let observer;
  const describe = id => `${id} spectrograms, top to bottom: recorded speech, Snake with MPD+MRD, Snake with MRD-only, LeakyReLU with MPD+MRD, and LeakyReLU with MRD-only. Shared frequency range 0 to 12 kHz and magnitude scale minus 100 to 0 dB relative to 1.`;
  const updateLabels = () => {
    const utterance = data.utterances[select.selectedIndex];
    previous.disabled = select.selectedIndex === 0;
    next.disabled = select.selectedIndex === data.utterances.length - 1;
    document.getElementById("atlas-info").textContent = `${select.selectedIndex + 1} / ${data.utterances.length} · ${utterance.reference_id}`;
    document.getElementById("atlas-listen").textContent = `Listen to ${utterance.id}`;
    document.getElementById("atlas-caption").textContent = `${utterance.id} · Identical analysis and absolute magnitude scale across the five conditions. These exploratory illustrations do not themselves establish perceptual quality or statistical significance.`;
  };
  const show = () => {
    observer?.disconnect();
    const id = select.value;
    const current = ++request;
    updateLabels();
    image.hidden = true;
    retry.hidden = true;
    viewport.setAttribute("aria-busy", "true");
    status.textContent = `Loading ${id} spectrograms…`;
    const candidate = new Image();
    candidate.decoding = "async";
    candidate.onload = () => {
      if (current !== request) return;
      image.src = candidate.src;
      image.alt = describe(id);
      image.hidden = false;
      viewport.removeAttribute("aria-busy");
      status.textContent = "";
    };
    candidate.onerror = () => {
      if (current !== request) return;
      viewport.removeAttribute("aria-busy");
      status.textContent = `Could not load ${id}. Check your connection and retry.`;
      retry.hidden = false;
    };
    candidate.src = `figures/spectrograms/${id}.webp`;
  };
  select.addEventListener("change", show);
  previous.addEventListener("click", () => { if (select.selectedIndex > 0) { select.selectedIndex--; show(); } });
  next.addEventListener("click", () => { if (select.selectedIndex < data.utterances.length - 1) { select.selectedIndex++; show(); } });
  retry.addEventListener("click", show);
  document.getElementById("atlas-enlarge").addEventListener("click", event => {
    const enlarged = viewport.classList.toggle("is-enlarged");
    event.currentTarget.setAttribute("aria-pressed", String(enlarged));
    event.currentTarget.textContent = enlarged ? "Standard size" : "Enlarge";
  });
  document.getElementById("atlas-listen").addEventListener("click", () => {
    const audioSelect = document.getElementById("utterance");
    if (audioSelect) {
      audioSelect.value = select.value;
      audioSelect.dispatchEvent(new Event("change", {bubbles:true}));
    }
  });
  document.getElementById("atlas-controls").hidden = false;
  updateLabels();
  // Keep the initial page light: fetch this atlas page only as its viewer approaches.
  if ("IntersectionObserver" in window) {
    observer = new IntersectionObserver(entries => { if (entries.some(entry => entry.isIntersecting)) show(); }, {rootMargin:"300px"});
    observer.observe(viewport);
  } else {
    show();
  }
})();
