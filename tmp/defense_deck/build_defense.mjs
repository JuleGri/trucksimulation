import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "C:/Users/Jule/Documents/Master/Masterthesis/Data-Aware-Process-Simulation-at-CTB";
const BUILD = path.join(ROOT, "tmp", "defense_deck");
const RENDER = path.join(BUILD, "rendered");
const OUTPUT = path.join(ROOT, "output", "presentation", "Jule_Grigat_Master_Thesis_Defense.pptx");

const W = 1280;
const H = 720;
const C = {
  white: "#FFFFFF",
  ink: "#0B1220",
  muted: "#5F6B7A",
  faint: "#98A2B3",
  line: "#D0D5DD",
  panel: "#F2F4F7",
  panel2: "#F8FAFC",
  blue: "#2F80ED",
  blue2: "#EAF3FF",
  teal: "#138A8A",
  green: "#16876A",
  green2: "#EAF8F3",
  amber: "#C98200",
  amber2: "#FFF4D8",
  red: "#C8102E",
  red2: "#FDECEF",
  dark: "#111827",
};
const FONT = "Aptos";

function box(slide, x, y, w, h, fill = C.panel, line = C.line, radius = "rounded-xl") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
    borderRadius: radius,
  });
}

function rect(slide, x, y, w, h, fill) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

function textBox(slide, value, x, y, w, h, opts = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = value;
  s.text.style = {
    fontSize: opts.size ?? 25,
    typeface: FONT,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
  };
  return s;
}

function circle(slide, x, y, d, fill, line = fill) {
  return slide.shapes.add({
    geometry: "ellipse",
    position: { left: x, top: y, width: d, height: d },
    fill,
    line: { style: "solid", fill: line, width: 1 },
  });
}

function addBase(p, title, n, appendix = false) {
  const slide = p.slides.add();
  slide.background.fill = C.white;
  if (appendix) {
    textBox(slide, "APPENDIX", 48, 24, 160, 24, { size: 17, color: C.red, bold: true });
    textBox(slide, title, 48, 51, 1160, 60, { size: 44, bold: true });
  } else {
    textBox(slide, title, 48, 34, 1184, 72, { size: 46, bold: true });
  }
  rect(slide, 48, 681, 1130, 1, C.line);
  textBox(slide, "Jule Grigat · Master’s Thesis Defense", 48, 687, 480, 20, { size: 15, color: C.faint });
  textBox(slide, String(n).padStart(2, "0"), 1180, 687, 52, 20, { size: 15, color: C.faint, align: "right" });
  return slide;
}

function addNote(slide, talk, sources) {
  const note = `${talk}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`;
  slide.speakerNotes.textFrame.setText(note);
  slide.speakerNotes.setVisible(true);
}

function statCard(slide, x, y, w, h, value, label, tone = "blue", detail = "") {
  const fills = { blue: C.blue2, green: C.green2, amber: C.amber2, red: C.red2, gray: C.panel };
  const accents = { blue: C.blue, green: C.green, amber: C.amber, red: C.red, gray: C.ink };
  box(slide, x, y, w, h, fills[tone], "none");
  rect(slide, x, y, 7, h, accents[tone]);
  textBox(slide, value, x + 26, y + 20, w - 52, 62, { size: 46, color: accents[tone], bold: true });
  textBox(slide, label, x + 26, y + 84, w - 52, 56, { size: 24, bold: true });
  if (detail && h >= 180) textBox(slide, detail, x + 26, y + h - 54, w - 52, 38, { size: 18, color: C.muted });
}

function takeaway(slide, x, y, w, h, number, heading, body, tone = "blue") {
  const colors = { blue: C.blue, green: C.green, amber: C.amber, red: C.red };
  box(slide, x, y, w, h, C.panel2, C.line);
  circle(slide, x + 24, y + 24, 42, colors[tone]);
  textBox(slide, String(number), x + 24, y + 29, 42, 28, { size: 22, color: C.white, bold: true, align: "center" });
  if (!body) {
    textBox(slide, heading, x + 84, y + 10, w - 108, h - 20, { size: h < 100 ? 21 : 25, bold: true, valign: "middle" });
    return;
  }
  if (h < 150) {
    textBox(slide, heading, x + 84, y + 18, w - 108, 38, { size: 23, bold: true });
    textBox(slide, body, x + 84, y + 58, w - 108, h - 68, { size: 18, color: C.muted });
  } else {
    textBox(slide, heading, x + 84, y + 20, w - 108, 64, { size: 25, bold: true });
    textBox(slide, body, x + 84, y + 88, w - 108, h - 106, { size: 21, color: C.muted });
  }
}

function arrow(slide, x, y, w = 44) {
  textBox(slide, "→", x, y, w, 46, { size: 36, color: C.faint, bold: true, align: "center", valign: "middle" });
}

function ciPlot(slide, x, y, w, rows, min, max, accent = C.blue) {
  const labelW = 235;
  const valueW = 185;
  const plotX = x + labelW;
  const plotW = w - labelW - valueW;
  const px = (v) => plotX + ((v - min) / (max - min)) * plotW;
  const zeroX = px(0);
  rect(slide, plotX, y + 12, plotW, 2, C.line);
  for (const tick of [min, 0, max]) {
    const tx = px(tick);
    rect(slide, tx, y + 6, 2, 15, tick === 0 ? C.ink : C.line);
    textBox(slide, `${tick >= 0 ? "+" : ""}${tick.toFixed(2)}`, tx - 34, y + 22, 68, 25, { size: 16, color: C.muted, align: "center" });
  }
  const rowStart = y + 66;
  const rowGap = 108;
  rect(slide, zeroX, rowStart - 15, 2, rowGap * rows.length - 22, C.faint);
  rows.forEach((r, i) => {
    const cy = rowStart + i * rowGap;
    const resolved = r.lo > 0 || r.hi < 0;
    const color = resolved ? accent : C.blue;
    textBox(slide, r.label, x, cy - 25, labelW - 18, 60, { size: 21, bold: true, valign: "middle" });
    rect(slide, px(r.lo), cy - 2, Math.max(4, px(r.hi) - px(r.lo)), 4, color);
    rect(slide, px(r.lo) - 1, cy - 10, 2, 20, color);
    rect(slide, px(r.hi) - 1, cy - 10, 2, 20, color);
    circle(slide, px(r.value) - 8, cy - 8, 16, color);
    textBox(slide, `${r.value >= 0 ? "+" : ""}${r.value.toFixed(3)}  [${r.lo >= 0 ? "+" : ""}${r.lo.toFixed(3)}, ${r.hi >= 0 ? "+" : ""}${r.hi.toFixed(3)}]`, plotX + plotW + 15, cy - 20, valueW - 10, 42, { size: 18, color: resolved ? color : C.muted, bold: resolved, valign: "middle" });
  });
}

async function main() {
  await fs.mkdir(RENDER, { recursive: true });
  await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
  const [logo, flows, petri] = await Promise.all([
    fs.readFile(path.join(ROOT, "resources", "unipd-bn.png")),
    fs.readFile(path.join(ROOT, "figures", "background", "container_flows_hhla.png")),
    fs.readFile(path.join(ROOT, "figures", "discovery", "petri_net_inductive.png")),
  ]);

  const p = Presentation.create({ slideSize: { width: W, height: H } });

  // 01 — title
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    rect(s, 0, 0, 18, H, C.red);
    s.images.add({ blob: logo, contentType: "image/png", alt: "University of Padova seal", fit: "contain", position: { left: 1038, top: 42, width: 156, height: 156 } });
    textBox(s, "MASTER’S THESIS DEFENSE", 62, 54, 430, 34, { size: 20, color: C.red, bold: true });
    textBox(s, "Data-Aware Process Simulation\nunder Physical Constraints", 62, 135, 930, 180, { size: 59, bold: true });
    textBox(s, "A Container Terminal Truck Operations Case Study", 66, 332, 850, 52, { size: 31, color: C.muted });
    rect(s, 64, 426, 1080, 2, C.line);
    textBox(s, "Jule Grigat", 66, 458, 400, 44, { size: 30, bold: true });
    textBox(s, "MSc Data Science · University of Padova · 2026", 66, 507, 650, 36, { size: 22, color: C.muted });
    textBox(s, "Supervisor: Prof. Massimiliano di Leoni\nCo-advisors: Marcel Petersen · Prof. Esteban Zimanyi", 66, 568, 740, 62, { size: 18, color: C.muted });
    textBox(s, "20 min", 1056, 591, 112, 40, { size: 24, color: C.red, bold: true, align: "center" });
    addNote(s, "Opening — target 25 seconds. Good morning, and thank you for the opportunity to present my thesis. I evaluated whether an automatically discovered, data-aware process simulator can be transferred from event-log data to the physically constrained setting of container-terminal truck operations. I will focus on the results and their interpretation: what the model reproduces, what additional context changes, and what the scenario experiments reveal about trustworthy decision support.", ["Grigat, thesis title page and abstract."]);
  }

  // 02 — problem
  {
    const s = addBase(p, "The decision is physical; the data are event-based", 2);
    box(s, 48, 126, 710, 510, C.panel2, "none");
    s.images.add({ blob: flows, contentType: "image/png", alt: "HHLA schematic of waterside, yard and hinterland container flows", fit: "contain", position: { left: 70, top: 162, width: 666, height: 282 } });
    textBox(s, "Operational question", 82, 470, 250, 34, { size: 20, color: C.red, bold: true });
    textBox(s, "What happens to truck turnaround when demand or yard capacity changes?", 82, 510, 630, 88, { size: 29, bold: true });
    takeaway(s, 790, 132, 442, 138, 1, "Sequential visits", "One truck cannot perform two yard activities at once.", "blue");
    takeaway(s, 790, 292, 442, 138, 2, "Shared resources", "Blocks, handling equipment and traffic interact.", "amber");
    takeaway(s, 790, 452, 442, 138, 3, "Spatial propagation", "Location and movement determine whether a change creates delay.", "red");
    addNote(s, "Target 1 minute 10 seconds. The motivating decision is concrete: before changing truck-slot rates or a yard-resource policy, a terminal would like to estimate the effect on turnaround. But the operational process is physical. A truck follows a sequential route, shares resources with other demand, and moves through a spatial system. The available data, by contrast, are recorded events. That creates the central transfer problem: can an event-log-derived model represent enough of the terminal to support the intended decision?", ["Grigat, thesis Sections 2.1, 3.1, and 4.1.", "HHLA container-flow schematic reproduced in thesis Figure 2.1; local source: figures/background/container_flows_hhla.png."]);
  }

  // 03 — central question
  {
    const s = addBase(p, "The question is not ‘can it run?’ but ‘what can we trust?’", 3);
    textBox(s, "How far can a simulator discovered from historical CTB events reproduce unseen behaviour—and where does the abstraction stop?", 94, 150, 1092, 112, { size: 34, bold: true, align: "center" });
    const cols = [70, 447, 824];
    const items = [
      ["Historical fidelity", "Does the simulated later period resemble the real hold-out?", "RQ1", C.blue],
      ["Structural validity", "Does the discovered process respect the physical visit sequence?", "RQ2", C.green],
      ["Scenario credibility", "Do context and interventions produce interpretable, defensible responses?", "RQ3–RQ4", C.red],
    ];
    items.forEach((it, i) => {
      box(s, cols[i], 316, 336, 246, C.panel2, C.line);
      rect(s, cols[i], 316, 336, 8, it[3]);
      textBox(s, it[2], cols[i] + 24, 340, 90, 25, { size: 17, color: it[3], bold: true });
      textBox(s, it[0], cols[i] + 24, 380, 288, 45, { size: 27, bold: true });
      textBox(s, it[1], cols[i] + 24, 444, 284, 83, { size: 21, color: C.muted });
    });
    textBox(s, "These are separate tests: success in one does not imply success in the others.", 198, 598, 884, 34, { size: 23, color: C.muted, bold: true, align: "center" });
    addNote(s, "Target 1 minute. The thesis deliberately separates three claims that are often conflated. First, historical fidelity: can the simulator reproduce a later, unseen period? Second, structural validity: does the process model obey the domain contract that a truck visit is sequential? Third, scenario credibility: do richer context and interventions create responses that can be interpreted operationally? My main argument is that these tests produce different answers, so scenario executability cannot be treated as evidence of physical validity.", ["Grigat, thesis Sections 1.2–1.3 and Chapter 6."]);
  }

  // 04 — evaluation design
  {
    const s = addBase(p, "One temporal hold-out separates discovery from validation", 4);
    const xs = [55, 354, 653, 952];
    const panels = [
      ["01", "Operational data", "89,460 visits\n275,408 events\n2 continuous months", C.blue],
      ["02", "Temporal split", "71,568 train\n17,892 held out\nlater period only", C.green],
      ["03", "Repeated simulation", "10 seeds (42–51)\n17,892 cases/run\nfrozen model", C.amber],
      ["04", "Paired scenarios", "T22 reallocation\n+20% working-time\narrival intensity", C.red],
    ];
    panels.forEach((it, i) => {
      box(s, xs[i], 169, 270, 354, C.panel2, C.line);
      circle(s, xs[i] + 22, 190, 46, it[3]);
      textBox(s, it[0], xs[i] + 22, 200, 46, 25, { size: 18, color: C.white, bold: true, align: "center" });
      textBox(s, it[1], xs[i] + 22, 260, 226, 64, { size: 27, bold: true });
      textBox(s, it[2], xs[i] + 22, 347, 226, 130, { size: 22, color: C.muted });
      if (i < 3) arrow(s, xs[i] + 270, 320, 30);
    });
    textBox(s, "Calibration uses training data only; the hold-out is untouched until evaluation.", 168, 567, 944, 40, { size: 24, color: C.ink, bold: true, align: "center" });
    addNote(s, "Target 1 minute. The evidence comes from 89,460 truck visits and 275,408 events over two continuous months. The first 80 percent of cases are used for discovery and calibration; the later 17,892 cases are held out. This temporal split preserves period shift instead of mixing it into both partitions. Each final comparison uses ten simulation seeds, and the two scenarios are evaluated with matched seeds against the same frozen baseline. I will not spend time on event-log engineering here; the key point is that all calibration choices are fixed before the hold-out evaluation.", ["Grigat, thesis Sections 3.6, 4.4–4.6, and 5.1."]);
  }

  // 05 — process structure
  {
    const s = addBase(p, "The process structure transfers—if event order is protected", 5);
    box(s, 48, 126, 795, 515, C.panel2, "none");
    s.images.add({ blob: petri, contentType: "image/png", alt: "Discovered Inductive-Miner Petri net for CTB truck visits", fit: "contain", position: { left: 70, top: 164, width: 750, height: 402 } });
    textBox(s, "Gate In → variable sequential yard loop → Gate Out", 114, 578, 664, 34, { size: 22, color: C.muted, bold: true, align: "center" });
    statCard(s, 874, 130, 350, 146, "99.994%", "held-out traces fit completely", "green");
    statCard(s, 874, 298, 350, 146, "0.9413", "held-out generalisation", "blue");
    statCard(s, 874, 466, 350, 146, "0", "parallel operators", "red", "7 places · 16 transitions · 32 arcs");
    addNote(s, "Target 1 minute 20 seconds. The control-flow result is strong, but it depends on preserving the authoritative event order during conversion. Once that order is protected, the Inductive Miner produces a compact model with no parallel operator. It fits 17,891 of 17,892 held-out cases completely and generalises beyond memorised training variants. Precision is lower because the loop allows additional sequential yard combinations, but that is different from forbidden concurrency. So the discovered net is suitable as the simulation backbone; the important methodological lesson is to validate the operator semantics, not just a global fitness score.", ["Grigat, thesis Section 5.2 and Table 5.2.", "Local figure: figures/discovery/petri_net_inductive.png."]);
  }

  // 06 — headline fidelity
  {
    const s = addBase(p, "Historical fidelity is strong locally, weak end-to-end", 6);
    textBox(s, "The same frozen simulator gives three very different answers.", 48, 118, 850, 40, { size: 25, color: C.muted });
    statCard(s, 55, 196, 366, 292, "0.072 min", "Inter-arrival EMD", "green", "Arrival spacing is reproduced closely.");
    statCard(s, 457, 196, 366, 292, "2.366 min", "Weighted yard-service EMD", "blue", "Common activities dominate the aggregate.");
    statCard(s, 859, 196, 366, 292, "8.180 min", "Turnaround EMD", "red", "The complete visit remains biased.");
    box(s, 182, 530, 916, 86, C.panel2, "none");
    textBox(s, "Interpretation", 210, 549, 150, 26, { size: 18, color: C.red, bold: true });
    textBox(s, "Accurate components do not guarantee an accurate system-level KPI.", 366, 544, 696, 42, { size: 27, bold: true });
    addNote(s, "Target 1 minute 20 seconds. RQ1 produces a mixed result. Inter-arrival spacing is reproduced extremely closely, with an EMD of 0.072 minutes. Weighted yard-service timing is also reasonably close at 2.366 minutes. But turnaround EMD is 8.180 minutes. This is important because these are not three different models; they are three views of the same baseline. The model can reproduce local distributions while still missing time that accumulates between or around recorded activities. Historical fidelity is therefore multidimensional, and the operational KPI must be evaluated directly.", ["Grigat, thesis Section 5.3 and Table 5.3."]);
  }

  // 07 — turnaround bias chart
  {
    const s = addBase(p, "Turnaround is systematically too short", 7);
    box(s, 48, 128, 730, 500, C.panel2, "none");
    s.charts.add("bar", {
      position: { left: 78, top: 164, width: 670, height: 410 },
      categories: ["Mean", "P90"],
      series: [
        { name: "Real hold-out", values: [39.742, 71.0], fill: C.ink },
        { name: "Simulation", values: [31.572, 53.69], fill: C.blue },
      ],
      hasLegend: true,
      legend: { position: "bottom", overlay: false },
      dataLabels: { showValue: true, position: "outEnd" },
      chartFill: C.panel2,
      chartLine: { style: "solid", width: 0, fill: C.panel2 },
      plotAreaFill: { type: "none" },
      plotAreaLine: { style: "solid", width: 0, fill: C.panel2 },
      xAxis: { visible: true, deleted: false, line: { style: "solid", width: 1, fill: C.line }, textStyle: { typeface: FONT, fontSize: "20px", color: C.ink } },
      yAxis: { visible: true, deleted: false, min: 0, max: 80, majorUnit: 20, majorGridlines: { style: "solid", width: 1, fill: C.line }, line: { style: "solid", width: 0, fill: C.panel2 }, textStyle: { typeface: FONT, fontSize: "17px", color: C.muted }, title: { text: "minutes" } },
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 70 },
    });
    statCard(s, 820, 134, 404, 195, "−8.17 min", "mean turnaround bias", "red", "−20.56% versus the hold-out");
    statCard(s, 820, 354, 404, 195, "−17.31 min", "P90 turnaround bias", "red", "−24.38% in the upper tail");
    textBox(s, "Narrow seed intervals show stable simulation output—not a correct model.", 820, 570, 404, 56, { size: 21, color: C.muted, bold: true });
    addNote(s, "Target 1 minute 30 seconds. The operational weakness becomes clear in the summary statistics. Real mean turnaround is 39.74 minutes, while the simulation gives 31.57: an 8.17-minute or 20.6 percent under-prediction. At the 90th percentile, the gap grows to 17.31 minutes. This bias persists across all ten seeds. The intervals around the simulated means are narrow, but that only tells us that random simulation variation is small. It does not make the frozen model correct. The missing time is systematic and points to structural rather than Monte-Carlo uncertainty.", ["Grigat, thesis Section 5.3, Table 5.3, and Section 5.9."]);
  }

  // 08 — configuration context
  {
    const s = addBase(p, "More context does not uniformly improve fidelity", 8);
    textBox(s, "no-rules", 109, 131, 200, 35, { size: 22, color: C.muted, bold: true, align: "center" });
    textBox(s, "rules + workload", 401, 131, 250, 35, { size: 22, color: C.blue, bold: true, align: "center" });
    const rows = [
      ["Turnaround EMD", "7.996", "8.131", "+0.135 min", "red"],
      ["Weighted service EMD", "2.360", "2.354", "−0.006 min", "green"],
      ["Activity-rate L1 error", "0.1753", "0.2264", "+29%", "red"],
      ["Gate-only cases / run", "11.9", "0.0", "−100%", "green"],
    ];
    rows.forEach((r, i) => {
      const yy = 184 + i * 104;
      box(s, 48, yy, 710, 82, i % 2 === 0 ? C.panel2 : C.white, "none");
      textBox(s, r[0], 70, yy + 23, 270, 34, { size: 22, bold: true });
      textBox(s, r[1], 344, yy + 18, 130, 42, { size: 28, color: C.muted, bold: true, align: "center" });
      textBox(s, "→", 485, yy + 18, 50, 42, { size: 29, color: C.faint, bold: true, align: "center" });
      textBox(s, r[2], 542, yy + 18, 130, 42, { size: 28, color: C.blue, bold: true, align: "center" });
      const tone = r[4] === "red" ? C.red : C.green;
      textBox(s, r[3], 671, yy + 22, 74, 32, { size: 18, color: tone, bold: true, align: "right" });
    });
    box(s, 802, 158, 420, 454, C.panel2, C.line);
    textBox(s, "What changed?", 834, 188, 340, 40, { size: 28, bold: true });
    takeaway(s, 826, 248, 372, 115, 1, "Timing: essentially tied", "Overlapping intervals; no practical timing advantage.", "blue");
    takeaway(s, 826, 369, 372, 115, 2, "Routing: cleaner", "The rare silent yard bypass disappears.", "green");
    takeaway(s, 826, 490, 372, 115, 3, "Incidence: worse", "Activity frequencies move further from reality.", "red");
    addNote(s, "Target 1 minute 25 seconds. RQ3 compares the distribution-only endpoint with the full rules-plus-workload endpoint after the same calibration. Timing is essentially tied: turnaround EMD becomes 0.135 minutes worse, while weighted service EMD improves by only 0.006 minutes. The clearer differences point in opposite directions. Context eliminates the rare gate-only bypass, but activity-rate error worsens from 0.1753 to 0.2264. The justified conclusion is not that data-aware modelling fails. It is that extra expressiveness changes different fidelity dimensions differently and must earn its complexity on held-out data.", ["Grigat, thesis Section 5.4 and Table 5.5."]);
  }

  // 09 — signal
  {
    const s = addBase(p, "Context works where the signal exists", 9);
    box(s, 55, 145, 560, 420, C.green2, "none");
    textBox(s, "ARRIVALS", 84, 174, 160, 25, { size: 18, color: C.green, bold: true });
    textBox(s, "Strong, stable calendar signal", 84, 216, 470, 50, { size: 30, bold: true });
    statCard(s, 84, 290, 225, 202, "0.762", "time-aware R²", "green");
    statCard(s, 333, 290, 252, 202, "−67.5%", "arrival-count MAE", "green", "day rate ≈ 8.7× night");
    box(s, 665, 145, 560, 420, C.red2, "none");
    textBox(s, "DOMINANT RMG DURATIONS", 694, 174, 310, 25, { size: 18, color: C.red, bold: true });
    textBox(s, "Available attributes explain almost nothing", 694, 216, 470, 80, { size: 30, bold: true });
    statCard(s, 694, 315, 225, 177, "−0.005", "RMG receive R²", "red");
    statCard(s, 943, 315, 252, 177, "−0.004", "RMG delivery R²", "red", ">60% of yard events are RMG");
    box(s, 175, 590, 930, 60, C.panel2, "none");
    textBox(s, "Data awareness helps only when information is available at the decision point and remains predictive over time.", 200, 604, 880, 38, { size: 23, bold: true, align: "center" });
    addNote(s, "Target 1 minute 20 seconds. The diagnostic explains why the aggregate RQ3 result is mixed. Calendar context is genuinely informative for arrivals: a time-aware tree reaches an R-squared of 0.762 and cuts arrival-count MAE by 67.5 percent; daytime intensity is about 8.7 times the night rate. In contrast, the available case-level attributes have slightly negative held-out R-squared for the two dominant RMG duration models, even though RMG accounts for more than 60 percent of yard events. Context is therefore useful selectively. More columns do not help when they do not measure the state that drives the outcome.", ["Grigat, thesis Sections 5.5–5.6 and Tables 5.6–5.8."]);
  }

  // 10 — observed patterns vs mechanisms
  {
    const s = addBase(p, "The model learns observed patterns—not missing mechanisms", 10);
    textBox(s, "Represented in the log", 95, 144, 440, 40, { size: 28, color: C.green, bold: true, align: "center" });
    textBox(s, "Required for a physical counterfactual", 745, 144, 440, 40, { size: 28, color: C.red, bold: true, align: "center" });
    const left = ["event order", "calendar pattern", "activity labels", "statistical resource IDs"];
    const right = ["explicit queue state", "truck & container location", "equipment movement", "block compatibility"];
    left.forEach((v, i) => {
      box(s, 110, 212 + i * 82, 395, 58, C.green2, "none");
      circle(s, 130, 228 + i * 82, 26, C.green);
      textBox(s, "✓", 130, 230 + i * 82, 26, 22, { size: 18, color: C.white, bold: true, align: "center" });
      textBox(s, v, 174, 227 + i * 82, 300, 30, { size: 23, bold: true });
    });
    right.forEach((v, i) => {
      box(s, 775, 212 + i * 82, 395, 58, C.red2, "none");
      circle(s, 795, 228 + i * 82, 26, C.red);
      textBox(s, "—", 795, 229 + i * 82, 26, 22, { size: 18, color: C.white, bold: true, align: "center" });
      textBox(s, v, 839, 227 + i * 82, 300, 30, { size: 23, bold: true });
    });
    arrow(s, 588, 335, 104);
    textBox(s, "gap", 608, 380, 64, 26, { size: 18, color: C.red, bold: true, align: "center" });
    textBox(s, "Interpretability makes associations auditable; it does not make them causal.", 195, 585, 890, 48, { size: 27, bold: true, align: "center" });
    addNote(s, "Target 1 minute 20 seconds. This is the interpretive bridge to the scenarios. The simulator can learn what the log contains: event order, calendar effects, activity labels and statistical resource assignments. It cannot learn states that were never represented: the actual queue, truck or container position, equipment movement, or block compatibility. A transparent white-box model is valuable because this gap is visible. But interpretability only makes an association inspectable; it does not turn the association into a causal mechanism or guarantee that an intervention will propagate physically.", ["Grigat, thesis Section 5.8 and Chapter 6."]);
  }

  // 11 — scenario A
  {
    const s = addBase(p, "What-if A: T22 reallocation executes—but not as a physical closure", 11);
    ciPlot(s, 50, 150, 810, [
      { label: "Mean turnaround", value: 0.092, lo: -0.016, hi: 0.200 },
      { label: "Mean RMG service", value: -0.019, lo: -0.101, hi: 0.063 },
      { label: "Delivery pre-service", value: 0.094, lo: 0.027, hi: 0.160 },
    ], -0.15, 0.25, C.red);
    box(s, 890, 142, 334, 454, C.panel2, C.line);
    textBox(s, "Intervention check", 918, 168, 280, 30, { size: 20, color: C.green, bold: true });
    textBox(s, "591.5 → 0", 918, 207, 280, 56, { size: 40, color: C.green, bold: true });
    textBox(s, "T22 assignments per run", 918, 267, 280, 34, { size: 20, color: C.muted });
    rect(s, 918, 322, 250, 2, C.line);
    textBox(s, "What the experiment means", 918, 350, 280, 36, { size: 24, bold: true });
    textBox(s, "Work is reassigned inside an abstract interchangeable resource pool. The model does not check location, relocation or travel.", 918, 404, 270, 134, { size: 22, color: C.muted });
    textBox(s, "Only delivery pre-service excludes zero.", 918, 548, 270, 42, { size: 20, color: C.red, bold: true });
    addNote(s, "Target 1 minute 25 seconds. In Scenario A, T22 is removed from the simulated RMG pool. The intervention executes exactly: assignments fall from an average of 591.5 to zero. Mean turnaround increases by only 0.092 minutes, and its paired interval includes zero. Delivery pre-service increases by 0.094 minutes and is the only activity-specific interval shown here that excludes zero. But this is not a physical block-closure study. The selector simply reallocates work to another statistical resource; it does not represent container location, relocation or added travel. The result is valid inside the model, not established for the terminal.", ["Grigat, thesis Sections 5.7.1 and Table 5.11."]);
  }

  // 12 — scenario B
  {
    const s = addBase(p, "What-if B: +20% demand barely moves turnaround", 12);
    ciPlot(s, 50, 150, 810, [
      { label: "Mean turnaround", value: 0.131, lo: -0.026, hi: 0.288 },
      { label: "Mean RMG service", value: 0.043, lo: -0.056, hi: 0.143 },
      { label: "Delivery pre-service", value: 0.153, lo: 0.041, hi: 0.265 },
    ], -0.10, 0.32, C.red);
    box(s, 890, 142, 334, 454, C.panel2, C.line);
    textBox(s, "Intervention check", 918, 168, 280, 30, { size: 20, color: C.green, bold: true });
    textBox(s, "+20.0%", 918, 207, 280, 56, { size: 40, color: C.green, bold: true });
    textBox(s, "working-time arrival intensity", 918, 267, 280, 46, { size: 20, color: C.muted });
    rect(s, 918, 328, 250, 2, C.line);
    textBox(s, "Diagnostic result", 918, 356, 280, 36, { size: 24, bold: true });
    textBox(s, "Arrivals change, but limited queueing and spatial interaction prevent a strong end-to-end delay response.", 918, 410, 270, 118, { size: 21, color: C.muted });
    textBox(s, "Turnaround CI includes zero.", 918, 540, 270, 46, { size: 20, color: C.red, bold: true });
    addNote(s, "Target 1 minute 25 seconds. Scenario B increases the model’s working-time arrival intensity by exactly 20 percent. Yet mean turnaround changes by only 0.131 minutes, with an interval that includes zero. Delivery pre-service increases by 0.153 minutes—about 9.2 seconds—and excludes zero, while the broader RMG measures do not. This is not evidence that the real terminal can absorb 20 percent more demand. It is a diagnostic result: the model changes arrival timing, but it lacks enough explicit queueing, spatial state and equipment interaction to create a credible congestion response.", ["Grigat, thesis Sections 5.7.2 and Table 5.11."]);
  }

  // 13 — scenario interpretation
  {
    const s = addBase(p, "The scenarios are stress tests of model semantics", 13);
    const xs = [78, 459, 840];
    const steps = [
      ["1", "Intervention executes", "T22 assignments vanish; the arrival parameter rises exactly 20%.", C.green],
      ["2", "Outputs are reproducible", "Matched seeds resolve small model-internal changes.", C.blue],
      ["3", "Physical response is absent", "Near-null turnaround exposes missing propagation mechanisms.", C.red],
    ];
    steps.forEach((st, i) => {
      box(s, xs[i], 180, 340, 282, C.panel2, C.line);
      circle(s, xs[i] + 26, 208, 48, st[3]);
      textBox(s, st[0], xs[i] + 26, 219, 48, 24, { size: 20, color: C.white, bold: true, align: "center" });
      textBox(s, st[1], xs[i] + 26, 286, 288, 70, { size: 29, bold: true });
      textBox(s, st[2], xs[i] + 26, 372, 288, 70, { size: 21, color: C.muted });
      if (i < 2) arrow(s, xs[i] + 340, 294, 41);
    });
    box(s, 142, 514, 996, 105, C.red2, "none");
    textBox(s, "A null effect is operationally meaningful only when the mechanisms that should propagate the intervention are represented and validated.", 174, 535, 932, 70, { size: 26, color: C.ink, bold: true, align: "center" });
    addNote(s, "Target 1 minute 15 seconds. The two scenarios pass an executability test and a reproducibility test. The intended parameters change, and matched seeds make the resulting model-internal deltas precise. But they fail a stronger physical interpretation test. A near-null outcome can be an operational finding only when the mechanism through which the intervention should act is represented and validated. Here the null turnaround responses instead reveal missing spatial and queueing propagation. In that sense, the scenarios are successful stress tests of model semantics, even though they are not sufficient operational forecasts.", ["Grigat, thesis Sections 5.7–5.8 and Chapter 6."]);
  }

  // 14 — decision support ladder
  {
    const s = addBase(p, "Decision support requires three distinct levels of evidence", 14);
    const levels = [
      ["1", "Historical reproduction", "PARTIAL", "Arrivals and common service times are close; turnaround is biased.", C.amber],
      ["2", "Executable white-box scenarios", "YES", "Rules and arrival parameters can be changed, inspected and repeated.", C.green],
      ["3", "Credible physical counterfactual", "NOT YET", "Spatial state, queue dynamics and equipment movement are missing.", C.red],
    ];
    levels.forEach((l, i) => {
      const y = 150 + i * 160;
      box(s, 74 + i * 55, y, 1095 - i * 110, 128, i === 0 ? C.amber2 : i === 1 ? C.green2 : C.red2, "none");
      circle(s, 98 + i * 55, y + 34, 56, l[4]);
      textBox(s, l[0], 98 + i * 55, y + 48, 56, 26, { size: 22, color: C.white, bold: true, align: "center" });
      textBox(s, l[1], 180 + i * 55, y + 24, 410, 42, { size: 29, bold: true });
      textBox(s, l[2], 875 - i * 15, y + 26, 210, 34, { size: 23, color: l[4], bold: true, align: "right" });
      textBox(s, l[3], 180 + i * 55, y + 72, 790 - i * 80, 42, { size: 21, color: C.muted });
    });
    addNote(s, "Target 1 minute 20 seconds. The findings can be summarised as a three-level evidence ladder. Historical reproduction is partial: the model captures important patterns but under-predicts the main operational KPI. Executable white-box scenarios are clearly supported: interventions are inspectable and repeatable. A credible physical counterfactual is not yet supported because the mechanisms that connect demand and resource changes to delay are incomplete. This distinction is the thesis’s practical contribution: it defines what the model can defensibly support today and what evidence is still required before prescriptive use.", ["Grigat, thesis Chapter 6."]);
  }

  // 15 — closing
  {
    const s = addBase(p, "The value is knowing where the model stops", 15);
    takeaway(s, 64, 151, 366, 252, 1, "Structure transfers", "The sequential process model generalises well when event order is protected.", "green");
    takeaway(s, 457, 151, 366, 252, 2, "Fidelity is multidimensional", "Strong local fits coexist with an 8.17-minute mean turnaround deficit.", "amber");
    takeaway(s, 850, 151, 366, 252, 3, "Executability is not causality", "Transparent scenarios reveal missing mechanisms; they do not prove terminal effects.", "red");
    textBox(s, "Data-aware simulation is a useful, auditable diagnostic—\nbut physical decision support needs richer state and hybrid mechanisms.", 116, 466, 1048, 92, { size: 33, bold: true, align: "center" });
    box(s, 486, 590, 308, 58, C.ink, "none");
    textBox(s, "Questions", 486, 602, 308, 34, { size: 25, color: C.white, bold: true, align: "center" });
    addNote(s, "Target 55 seconds. I would close with three points. First, automatically discovered structure can transfer to this logistics setting, but event-order semantics must be protected. Second, fidelity is multidimensional: strong component-level fits do not compensate for a systematic turnaround deficit. Third, scenario executability is not counterfactual validity. The model is already useful as a transparent and auditable diagnostic, especially because it makes its boundary visible. Moving from diagnosis to prescriptive terminal decisions requires richer event-level and spatial state plus explicit queueing and equipment mechanisms. I welcome your questions.", ["Grigat, thesis Chapter 6."]);
  }

  // 16 — appendix RQs
  {
    const s = addBase(p, "Research questions and one-sentence answers", 16, true);
    const qs = [
      ["RQ1", "Historical fidelity", "Partial: arrivals and service timing transfer better than full turnaround."],
      ["RQ2", "Process structure", "Suitable backbone after event-order protection; no parallel operator."],
      ["RQ3", "Contextual expressiveness", "Changes fidelity dimensions in different directions; no uniform improvement."],
      ["RQ4", "What-if decision support", "Experiments are transparent, but physical counterfactuals are not established."],
    ];
    qs.forEach((q, i) => {
      const y = 132 + i * 126;
      box(s, 66, y, 1148, 102, i % 2 === 0 ? C.panel2 : C.white, C.line);
      textBox(s, q[0], 88, y + 24, 84, 34, { size: 23, color: i === 3 ? C.red : C.blue, bold: true });
      textBox(s, q[1], 188, y + 20, 330, 38, { size: 26, bold: true });
      textBox(s, q[2], 532, y + 20, 642, 58, { size: 22, color: C.muted });
    });
    addNote(s, "Use this slide when an examiner asks you to restate the research design. Answer each question directly before adding nuance. The most important connective sentence is: the four answers are deliberately non-equivalent—structural validity, historical fidelity and scenario credibility must be evaluated separately.", ["Grigat, thesis Section 1.3 and Chapter 6."]);
  }

  // 17 — appendix case/event log
  {
    const s = addBase(p, "What exactly is represented in the event log?", 17, true);
    statCard(s, 58, 134, 352, 172, "89,460", "truck visits = cases", "blue");
    statCard(s, 464, 134, 352, 172, "275,408", "activity events", "green");
    statCard(s, 870, 134, 352, 172, "2 months", "continuous CTB data", "amber");
    box(s, 58, 342, 748, 266, C.panel2, C.line);
    textBox(s, "Case semantics", 84, 370, 250, 36, { size: 27, bold: true });
    textBox(s, "Gate In", 100, 445, 130, 50, { size: 23, bold: true, align: "center", valign: "middle" });
    arrow(s, 232, 447, 50);
    textBox(s, "1+ yard activities", 292, 445, 220, 50, { size: 23, bold: true, align: "center", valign: "middle" });
    arrow(s, 515, 447, 50);
    textBox(s, "Gate Out", 575, 445, 130, 50, { size: 23, bold: true, align: "center", valign: "middle" });
    textBox(s, "A case is one truck visit; its authoritative XES order is sequential.", 100, 530, 606, 42, { size: 21, color: C.muted, align: "center" });
    box(s, 838, 342, 384, 266, C.red2, "none");
    textBox(s, "Important timestamp limit", 865, 370, 330, 40, { size: 25, color: C.red, bold: true });
    textBox(s, "The real log has start and complete timestamps, but no enablement timestamp. Simulated pre-service delay is therefore not validated real queueing time.", 865, 435, 320, 132, { size: 22, color: C.ink });
    addNote(s, "If asked about event-log engineering, keep the answer conceptual. A case is one truck visit from Gate In through one or more yard activities to Gate Out. The real log supports observed service time and turnaround, but not an observed enablement-to-start queue measure. That is why the thesis consistently calls the simulated quantity pre-service delay and avoids claiming it is validated queueing time.", ["Grigat, thesis Chapter 3 and Section 5.4, ‘Meaning of the Time Measures’."]);
  }

  // 18 — appendix protocol
  {
    const s = addBase(p, "Why temporal validation and ten seeds?", 18, true);
    const items = [
      ["Temporal 80/20 split", "Preserves later-period shifts; a random split would leak them into both partitions.", C.green],
      ["Training-only calibration", "The final resource cap and simulator choices are fixed before hold-out evaluation.", C.blue],
      ["10 seeds (42–51)", "Quantifies Monte-Carlo variation for one frozen model and one hold-out.", C.amber],
      ["Matched scenario seeds", "Reduces noise in scenario-minus-baseline changes.", C.red],
    ];
    items.forEach((it, i) => takeaway(s, 64 + (i % 2) * 592, 138 + Math.floor(i / 2) * 238, 548, 198, i + 1, it[0], it[1], i === 0 ? "green" : i === 1 ? "blue" : i === 2 ? "amber" : "red"));
    box(s, 237, 624, 806, 40, C.panel2, "none");
    textBox(s, "The confidence intervals do not include discovery, period or structural uncertainty.", 252, 632, 776, 25, { size: 20, color: C.red, bold: true, align: "center" });
    addNote(s, "A concise answer: the temporal split tests transfer to a genuinely later period. Ten seeds are enough to show that the reported simulation means are stable for the frozen model. They do not quantify all uncertainty. More seeds would narrow Monte-Carlo intervals, but they would not address another training period, another discovery run, or missing spatial state.", ["Grigat, thesis Sections 4.4–4.6, 5.1, and 5.9."]);
  }

  // 19 — appendix metrics
  {
    const s = addBase(p, "How to interpret the validation metrics", 19, true);
    const metrics = [
      ["EMD / Wasserstein", "Average distributional shift in the metric’s units; lower is better."],
      ["KS statistic", "Maximum CDF separation; lower is better, but it has no time unit."],
      ["Fitness", "Can the model replay observed traces? High fitness does not imply precise timing."],
      ["Precision", "How much extra behaviour does the model allow? Lower precision can reflect deliberate generalisation."],
      ["Generalisation", "Does the model support plausible unseen combinations rather than memorising variants?"],
      ["Activity-rate L1", "Absolute mismatch in yard-activity incidence per case; lower is better."],
    ];
    metrics.forEach((m, i) => {
      const y = 126 + i * 84;
      box(s, 60, y, 1160, 66, i % 2 === 0 ? C.panel2 : C.white, "none");
      textBox(s, m[0], 82, y + 17, 285, 32, { size: 22, bold: true });
      textBox(s, m[1], 386, y + 15, 800, 38, { size: 21, color: C.muted });
    });
    addNote(s, "The defense point is that no single metric establishes simulator validity. Fitness evaluates replay, precision evaluates permitted language, and distributional distances evaluate different timing outcomes. The thesis therefore combines metrics with explicit structural checks and operational KPI errors.", ["Grigat, thesis Sections 2.7 and 4.5."]);
  }

  // 20 — appendix control flow
  {
    const s = addBase(p, "Control-flow evidence: fitness is not the whole argument", 20, true);
    statCard(s, 62, 137, 352, 183, "0.999992", "held-out fitness", "green");
    statCard(s, 464, 137, 352, 183, "0.7558", "held-out precision", "amber");
    statCard(s, 866, 137, 352, 183, "0.9413", "held-out generalisation", "blue");
    box(s, 62, 360, 548, 246, C.panel2, C.line);
    textBox(s, "Why lower precision is acceptable here", 90, 388, 490, 38, { size: 26, bold: true });
    textBox(s, "The yard loop permits unseen sequential combinations and repetitions. The domain forbids concurrency—not every unseen order.", 90, 451, 480, 114, { size: 22, color: C.muted });
    box(s, 662, 360, 556, 246, C.red2, "none");
    textBox(s, "Disclosed overgeneralisation", 690, 388, 500, 38, { size: 26, color: C.red, bold: true });
    textBox(s, "A silent gate-only path remains formally possible. It appears in 0.067% of no-rules runs and in none of the context-aware runs; it is reported rather than hidden by an undocumented guard.", 690, 447, 488, 128, { size: 21, color: C.ink });
    addNote(s, "If challenged on precision, say that the model’s broader language is semantically inspected. It contains no parallel operator and produces no within-case overlap, so lower precision represents additional sequential combinations rather than prohibited concurrency. The silent bypass is the remaining auditable exception and is explicitly disclosed.", ["Grigat, thesis Section 5.2 and Table 5.2."]);
  }

  // 21 — appendix screening table
  {
    const s = addBase(p, "Full held-out configuration screening", 21, true);
    textBox(s, "Metric", 70, 122, 400, 28, { size: 21, color: C.muted, bold: true });
    textBox(s, "no-rules", 552, 122, 210, 28, { size: 21, color: C.muted, bold: true, align: "center" });
    textBox(s, "rules + workload", 842, 122, 270, 28, { size: 21, color: C.blue, bold: true, align: "center" });
    const rows = [
      ["Turnaround EMD (min)", "7.996 [7.878, 8.114]", "8.131 [8.063, 8.198]"],
      ["Turnaround KS", "0.1276 [0.1238, 0.1314]", "0.1331 [0.1298, 0.1363]"],
      ["Mean turnaround (min)", "31.747 [31.629, 31.865]", "31.617 [31.550, 31.685]"],
      ["P90 turnaround (min)", "54.400 [54.031, 54.769]", "53.600 [53.231, 53.969]"],
      ["Inter-arrival EMD (min)", "0.0765 [0.0696, 0.0835]", "0.0724 [0.0674, 0.0774]"],
      ["Weighted service EMD (min)", "2.360 [2.318, 2.401]", "2.354 [2.312, 2.396]"],
      ["Yard-activity-rate L1", "0.1753 [0.1710, 0.1795]", "0.2264 [0.2215, 0.2313]"],
      ["Gate-only cases / 17,892", "11.9 [9.6, 14.2]", "0.0 [0.0, 0.0]"],
    ];
    rows.forEach((r, i) => {
      const y = 161 + i * 56;
      box(s, 58, y, 1164, 48, i % 2 === 0 ? C.panel2 : C.white, "none");
      textBox(s, r[0], 76, y + 10, 410, 28, { size: 20, bold: true });
      textBox(s, r[1], 492, y + 10, 320, 28, { size: 19, color: C.muted, align: "center" });
      textBox(s, r[2], 822, y + 10, 378, 28, { size: 19, color: C.blue, align: "center" });
    });
    textBox(s, "No rules-only endpoint is available; the workload effect cannot be isolated.", 290, 626, 700, 30, { size: 20, color: C.red, bold: true, align: "center" });
    addNote(s, "Use this only if asked for the full comparison. Stress the missing middle configuration: this comparison identifies the combined effect of rules and workload features, not the isolated causal contribution of workload. That is why the conclusion is deliberately limited to ‘additional context changed fidelity dimensions in different directions.’", ["Grigat, thesis Section 5.4 and Table 5.5."]);
  }

  // 22 — appendix scenario table
  {
    const s = addBase(p, "Scenario details: paired changes and intervals", 22, true);
    box(s, 58, 126, 560, 492, C.panel2, C.line);
    textBox(s, "A · Remove T22 from RMG pool", 84, 152, 500, 42, { size: 28, bold: true });
    const a = [
      ["Mean turnaround", "+0.092 [−0.016, +0.200]"],
      ["P90 turnaround", "0.000 [−0.337, +0.337]"],
      ["Mean RMG service", "−0.019 [−0.101, +0.063]"],
      ["Mean RMG pre-service", "+0.065 [+0.019, +0.112]"],
      ["Delivery pre-service", "+0.094 [+0.027, +0.160]"],
      ["T22 assignments", "591.5 → 0.0 (exact)"],
    ];
    a.forEach((r, i) => {
      textBox(s, r[0], 86, 222 + i * 56, 235, 30, { size: 20, bold: true });
      textBox(s, r[1], 322, 222 + i * 56, 260, 30, { size: 19, color: i >= 3 ? C.red : C.muted, align: "right" });
    });
    box(s, 662, 126, 560, 492, C.panel2, C.line);
    textBox(s, "B · +20% working-time demand", 688, 152, 500, 42, { size: 28, bold: true });
    const b = [
      ["Working arrival rate", "+20.000% (exact)"],
      ["Elapsed-hour arrivals", "+15.910%"],
      ["Mean turnaround", "+0.131 [−0.026, +0.288]"],
      ["Mean RMG service", "+0.043 [−0.056, +0.143]"],
      ["Mean RMG pre-service", "+0.043 [−0.056, +0.142]"],
      ["Delivery pre-service", "+0.153 [+0.041, +0.265]"],
    ];
    b.forEach((r, i) => {
      textBox(s, r[0], 690, 222 + i * 56, 235, 30, { size: 20, bold: true });
      textBox(s, r[1], 926, 222 + i * 56, 260, 30, { size: 19, color: i === 5 ? C.red : C.muted, align: "right" });
    });
    textBox(s, "Intervals cover matched simulation-seed variation conditional on the frozen model.", 248, 635, 784, 28, { size: 20, color: C.muted, align: "center" });
    addNote(s, "This slide gives the exact scenario values. If asked why some small effects exclude zero, explain that paired seeds remove much of the Monte-Carlo noise. Then immediately qualify that statistical resolution conditional on one model is not the same as operational significance or structural validity.", ["Grigat, thesis Section 5.7 and Table 5.11."]);
  }

  // 23 — appendix validity/future
  {
    const s = addBase(p, "Threats to validity—and the path forward", 23, true);
    textBox(s, "Current boundary", 80, 129, 480, 40, { size: 29, color: C.red, bold: true });
    textBox(s, "Required extension", 720, 129, 480, 40, { size: 29, color: C.green, bold: true });
    const left = [
      "One terminal and one two-month window",
      "No observed enablement / queue timestamp",
      "Resource IDs do not encode physical feasibility",
      "Seed CIs exclude discovery and structural uncertainty",
    ];
    const right = [
      "Repeat discovery across seasons or terminals",
      "Add queue-entry, location and movement events",
      "Hybrid process + discrete-event / agent-based model",
      "Validate candidate policies against observed outcomes",
    ];
    left.forEach((v, i) => takeaway(s, 65, 190 + i * 103, 545, 82, i + 1, v, "", "red"));
    right.forEach((v, i) => takeaway(s, 670, 190 + i * 103, 545, 82, i + 1, v, "", "green"));
    addNote(s, "If asked what you would do next, answer in this order: richer state representation first, then a hybrid simulation formalism, then external validation, and finally policy validation. This shows that the proposed future work follows directly from the empirical failure modes rather than being a generic wish list.", ["Grigat, thesis Section 5.9 and Chapter 6, Future Work."]);
  }

  // 24 — appendix likely questions
  {
    const s = addBase(p, "Likely examiner questions: concise answers", 24, true);
    const qa = [
      ["Why not a random split?", "It would mix the later-period shift into training and test, overstating transfer."],
      ["Why use the context-aware model for scenarios?", "Its white-box rules can be inspected and changed; it was not selected as the best forecaster."],
      ["Does near-perfect fitness prove validity?", "No. It proves replay capability; timing and physical mechanisms require separate tests."],
      ["Why can demand rise without congestion?", "The simulator lacks explicit queue, location and equipment-interaction state."],
      ["Would you deploy it operationally?", "As an auditable diagnostic and experiment scaffold—yes; for prescriptive physical claims—not yet."],
      ["What is the main contribution?", "A reproducible empirical boundary between learned event-log patterns and absent physical mechanisms."],
    ];
    qa.forEach((q, i) => {
      const y = 120 + i * 89;
      textBox(s, q[0], 70, y + 10, 390, 54, { size: 22, bold: true });
      box(s, 472, y, 746, 70, i % 2 === 0 ? C.panel2 : C.white, "none");
      textBox(s, q[1], 496, y + 11, 694, 48, { size: 21, color: C.muted });
    });
    addNote(s, "Use these as 20-second answers: lead with the direct answer, then provide one supporting number or limitation. Avoid turning a question into a mini-lecture. If an examiner asks whether the model is useful, distinguish diagnostic usefulness from prescriptive readiness.", ["Grigat, thesis Chapters 5–6."]);
  }

  // Render all slides and export the editable PowerPoint.
  for (const [i, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    const png = await p.export({ slide, format: "png", scale: 1.5 });
    await fs.writeFile(path.join(RENDER, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(RENDER, `${stem}.layout.json`), await layout.text());
  }
  const montage = await p.export({ format: "webp", montage: true, scale: 0.5 });
  await fs.writeFile(path.join(BUILD, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUTPUT);
  console.log(`Wrote ${OUTPUT}`);
  console.log(`Rendered ${p.slides.items.length} slides to ${RENDER}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
