import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { Engine } from '../lib/engine.js';
import { defaultGliner, GlinerClassifier } from '../lib/gliner-adapter.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Initialize engines
const inTreeEngine = new Engine({ cascade: false, type: 'builtin' });
const glinerEngine = new Engine({ gliner: defaultGliner, type: 'gliner' });
const cascadeEngine = new Engine({ gliner: defaultGliner, cascade: true });

console.log('='.repeat(80));
console.log('EVALUATION: In-Tree Zero-Dep vs GLiNER 2.5 vs Speculative Cascade');
console.log('='.repeat(80));

// Part 1: Tough Edge Cases (Negation, Syntax, Context)
console.log('\n### Part 1: Hard Semantic Edge Cases (Where Heuristics Fail)\n');

const edgeCases = [
  {
    title: 'Negated intent with contrasting main clause',
    text: 'I do not want to cancel my subscription or get a refund, but the app crashed with 500 error',
    candidates: ['billing', 'bug'],
    expected: 'bug',
  },
  {
    title: 'Negated defect with actual billing request',
    text: 'The checkout button was not broken, but my credit card was charged twice for the same invoice',
    candidates: ['bug', 'billing'],
    expected: 'billing',
  },
  {
    title: 'Subtle sarcasm / indirect tone',
    text: 'Oh absolutely wonderful, another unexpected monthly subscription fee on my bank statement',
    candidates: ['praise', 'billing'],
    expected: 'billing',
  },
  {
    title: 'Negated spam trigger in legitimate email',
    text: 'Do not click on links promising free money or luxury gift prizes; this is a company security warning',
    candidates: ['spam', 'security'],
    expected: 'security',
  },
  {
    title: 'Paraphrased bug without keyword overlap',
    text: 'The user interface is completely frozen and unresponsive to any mouse clicks or keyboard input',
    candidates: ['bug', 'feature'],
    expected: 'bug',
  },
];

console.log('| Test Case | Expected | In-Tree (Heuristic) | GLiNER 2.5 | Cascade (In-Tree + GLiNER) |');
console.log('|---|---|---|---|---|');

for (const tc of edgeCases) {
  const rLocal = await inTreeEngine.classify(tc.text, tc.candidates);
  const rGliner = await glinerEngine.classify(tc.text, tc.candidates);
  const rCascade = await cascadeEngine.classify(tc.text, tc.candidates);

  const formatRes = (res, exp) => {
    const isOk = res.label === exp;
    const mark = isOk ? '✅' : '❌';
    const extra = res.fastPath ? ' (⚡L1)' : (res.engine === 'cascade' ? ' (🧠GLiNER)' : '');
    return `${mark} **${res.label}**${extra}`;
  };

  console.log(`| ${tc.title} | **${tc.expected}** | ${formatRes(rLocal, tc.expected)} | ${formatRes(rGliner, tc.expected)} | ${formatRes(rCascade, tc.expected)} |`);
}

// Part 2: Public Benchmark Datasets
console.log('\n\n### Part 2: Hugging Face Standard Benchmarks (N=100 per dataset)\n');

const benchmarks = [
  {
    name: 'Customer Intent Routing (PolyAI/banking77)',
    source: 'Hugging Face (mteb/banking77)',
    kind: 'choice',
    file: path.join(__dirname, 'banking77_10way.json'),
    useDescriptions: true,
  },
  {
    name: 'Spam Guardrails (ucirvine/sms_spam)',
    source: 'Hugging Face (ucirvine/sms_spam)',
    kind: 'noul',
    file: path.join(__dirname, 'sms_spam.json'),
  },
  {
    name: 'Topic Classification (fancyzhx/ag_news)',
    source: 'Hugging Face (fancyzhx/ag_news)',
    kind: 'choice',
    file: path.join(__dirname, 'ag_news.json'),
  },
  {
    name: 'Affective Emotion (dair-ai/emotion)',
    source: 'Hugging Face (dair-ai/emotion)',
    kind: 'choice',
    file: path.join(__dirname, 'emotion.json'),
  }
];

async function evaluateDataset(engine, data, b) {
  let correct = 0;
  let fastPaths = 0;
  const latencies = [];

  for (const item of data) {
    const t0 = performance.now();
    let res;
    if (b.kind === 'choice') {
      const labels = (b.useDescriptions && item.descriptions)
        ? item.candidates.map(c => `${c}: ${item.descriptions[c]}`)
        : item.candidates;
      res = await engine.classify(item.text, labels);
      const chosenKey = res.label.split(':')[0].trim().toLowerCase();
      const expectedKey = item.expected.toLowerCase();
      if (chosenKey === expectedKey) correct++;
    } else {
      res = await engine.predicate(item.text, item.condition);
      if (res.value === item.expected) correct++;
    }
    const t1 = performance.now();
    latencies.push(t1 - t0);
    if (res.fastPath) fastPaths++;
  }

  latencies.sort((a, b) => a - b);
  const meanLat = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const p95Lat = latencies[Math.floor(latencies.length * 0.95)] || 0;
  const acc = (correct / data.length) * 100;

  return {
    acc: `${acc.toFixed(1)}%`,
    meanLat: `${meanLat.toFixed(2)} ms`,
    p95Lat: `${p95Lat.toFixed(2)} ms`,
    fastPathRate: `${((fastPaths / data.length) * 100).toFixed(0)}%`,
  };
}

for (const b of benchmarks) {
  if (!fs.existsSync(b.file)) continue;
  const data = JSON.parse(fs.readFileSync(b.file, 'utf8'));

  console.log(`#### ${b.name}`);
  console.log('| Mode / Engine | Top-1 Accuracy | Mean Latency | p95 Latency | Fast-Path Rate |');
  console.log('|---|---|---|---|---|');

  const sInTree = await evaluateDataset(inTreeEngine, data, b);
  console.log(`| **In-Tree Pure JS (Heuristic)** | ${sInTree.acc} | ${sInTree.meanLat} | ${sInTree.p95Lat} | 100% (L1 cache) |`);

  const sCascade = await evaluateDataset(cascadeEngine, data, b);
  console.log(`| **Speculative Cascade (In-Tree + GLiNER)** | **${sCascade.acc}** | **${sCascade.meanLat}** | **${sCascade.p95Lat}** | **${sCascade.fastPathRate}** |`);
  console.log('');
}
