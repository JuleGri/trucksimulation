import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePptx = "C:/Users/Jule/Documents/Master/Masterthesis/Data-Aware-Process-Simulation-at-CTB/figures/schemas/CTB_pipeline_diagrams_black_white.pptx";
const finalPptx = sourcePptx;
const finalPng = "C:/Users/Jule/Documents/Master/Masterthesis/Data-Aware-Process-Simulation-at-CTB/figures/schemas/experimental_pipeline_bw.png";
const workDir = "C:/Users/Jule/Documents/Master/Masterthesis/Data-Aware-Process-Simulation-at-CTB/tmp/pipeline_pptx_edit";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function records(ndjson) {
  return ndjson
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

async function main() {
  await fs.mkdir(workDir, { recursive: true });
  await fs.copyFile(sourcePptx, `${workDir}/source-before-edit.pptx`);

  const presentation = await PresentationFile.importPptx(
    await FileBlob.load(sourcePptx),
  );

  const before = await presentation.inspect({
    kind: "slide,textbox,shape,notes,layout",
    maxChars: 30000,
  });
  await fs.writeFile(`${workDir}/before-inspect.ndjson`, before.ndjson);
  const beforeRecords = records(before.ndjson);
  const slide2Record = beforeRecords.find(
    (record) => record.kind === "slide" && record.slide === 2,
  );
  if (!slide2Record) throw new Error("Slide 2 was not found");
  const slide2 = presentation.resolve(slide2Record.id);

  await writeBlob(
    `${workDir}/before-slide-2.png`,
    await presentation.export({ slide: slide2, format: "png", scale: 2 }),
  );
  await fs.writeFile(
    `${workDir}/before-slide-2.layout.json`,
    await (await slide2.export({ format: "layout" })).text(),
  );

  const edits = [
    ["temporal-split", "Train / validation / test", "Temporal data split"],
    [
      "temporal-split-description",
      "64% discovery, 16% validation, 20% final test",
      "64% discovery, 16% validation, 20% final test",
    ],
    ["prefix-trie-description", "Use sequential net; verify held-out language and case contract", "Use sequential net and verify held-out language and case contract"],
    ["prosit-discovery-description", "2x2: static proxies off/on; native state off/on", "2x2: static proxies off/on, native state off/on"],
    ["freeze-calibrate", "Select and freeze models", "Select and freeze models"],
    [
      "freeze-calibrate-description",
      "Validate on 16%, refit on 80%, cap RMGs at three",
      "Validate on 16%, refit on 80%, cap RMGs at three",
    ],
    [
      "monte-carlo-description",
      "Ten seeds (42--51), 17,892 test cases per run",
      "Ten seeds (42--51), 17,892 test cases per run",
    ],
    ["held-out-evaluation", "Evaluate untouched test", "Evaluate untouched test"],
    [
      "what-if-scenarios-description",
      "Native-only: T22 closure and 20% higher working-time demand",
      "Native-only: T22 closure and 20% higher working-time demand",
    ],
  ];

  for (const [name, oldText, newText] of edits) {
    const record = beforeRecords.find((item) => item.name === name);
    if (!record) throw new Error(`Shape not found: ${name}`);
    const target = presentation.resolve(record.id);
    target.text.replace(oldText, newText);
  }

  const after = await presentation.inspect({
    kind: "slide,textbox,shape,notes,layout",
    maxChars: 30000,
  });
  await fs.writeFile(`${workDir}/after-inspect.ndjson`, after.ndjson);
  await writeBlob(
    `${workDir}/after-slide-2.png`,
    await presentation.export({ slide: slide2, format: "png", scale: 2 }),
  );
  await fs.writeFile(
    `${workDir}/after-slide-2.layout.json`,
    await (await slide2.export({ format: "layout" })).text(),
  );
  await writeBlob(
    `${workDir}/after-montage.webp`,
    await presentation.export({ format: "webp", montage: true, scale: 1 }),
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  await writeBlob(finalPng, await presentation.export({ slide: slide2, format: "png", scale: 2 }));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
