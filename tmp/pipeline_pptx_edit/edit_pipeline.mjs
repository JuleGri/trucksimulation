import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const root = "C:/Users/Jule/Documents/Master/Masterthesis/Data-Aware-Process-Simulation-at-CTB";
const tmp = `${root}/tmp/pipeline_pptx_edit`;
const input = `${tmp}/template-starter.pptx`;
const finalPptx = `${root}/figures/schemas/CTB_pipeline_diagrams_black_white.pptx`;
const finalPng = `${root}/figures/schemas/experimental_pipeline_bw.png`;

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(input));
const slide = presentation.slides.getItem(1);
const inventory = await presentation.inspect({
  kind: "slide,textbox,shape",
  maxChars: 20000,
});
const records = inventory.ndjson
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

function shapeByName(name) {
  const record = records.find((item) => item.slide === 2 && item.name === name);
  if (!record) throw new Error(`Missing inherited shape: ${name}`);
  return presentation.resolve(record.id);
}

await writeBlob(
  `${tmp}/before-slide-02.png`,
  await presentation.export({ slide, format: "png", scale: 2 }),
);
await fs.writeFile(
  `${tmp}/before-slide-02.layout.json`,
  await (await slide.export({ format: "layout" })).text(),
);

shapeByName("domain-audit-description").text.replace(
  "Parallel execution violates the sequential truck-visit contract",
  "Expert-validated: one truck cannot execute activities in parallel",
);
shapeByName("prefix-trie").text.replace(
  "Prefix-trie process skeleton",
  "Accepted Petri-net skeleton",
);
shapeByName("prefix-trie-description").text.replace(
  "Memorise training variants; check held-out coverage and case order",
  "Use sequential net; verify held-out language and case contract",
);
shapeByName("freeze-calibrate-description").text.replace(
  "Authoritative bundles; final RMG capacity cap of three",
  "Expert-validated RMG ceiling: three per block",
);
shapeByName("robustness-sensitivity-description").text.replace(
  "Configuration screening, arrivals, resources and transition percentiles",
  "Configuration, resource, transition and demand-saturation tests",
);

const after = await presentation.inspect({
  kind: "slide,textbox,shape,layout",
  search: "Accepted|Expert-validated|demand-saturation",
  maxChars: 8000,
});
await fs.writeFile(`${tmp}/after-inspect.ndjson`, after.ndjson, "utf8");

for (const [index, item] of presentation.slides.items.entries()) {
  const stem = `final-slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(
    `${tmp}/${stem}.png`,
    await presentation.export({ slide: item, format: "png", scale: 2 }),
  );
  await fs.writeFile(
    `${tmp}/${stem}.layout.json`,
    await (await item.export({ format: "layout" })).text(),
  );
}

const slidePng = await presentation.export({ slide, format: "png", scale: 2 });
await writeBlob(finalPng, slidePng);
await writeBlob(
  `${tmp}/final-montage.webp`,
  await presentation.export({ format: "webp", montage: true, scale: 1 }),
);

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(finalPptx);
