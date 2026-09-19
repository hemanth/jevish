import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { classifyZeroDep } from '../lib/zero-dep.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const benchmarks = [
  {
    name: 'AG News (4-way Topic Classification)',
    file: path.join(__dirname, 'ag_news.json'),
    baselines: [
      { name: 'hev (in-tree built-in)', engine: 'in-tree' },
      { name: 'Jev-1.13 (classifier.dev)', top1: '87.7%', meanLat: '~2.1 ms', p95Lat: '~3.4 ms', brier: '0.091', deps: 'System One' },
      { name: 'Fastino / GLiNER2.5', top1: '81.2%', meanLat: '~14.2 ms', p95Lat: '~22.5 ms', brier: '0.118', deps: 'Optional npm' },
      { name: 'SetFit / MiniLM-L6', top1: '74.6%', meanLat: '~18.0 ms', p95Lat: '~28.0 ms', brier: '0.142', deps: 'Python/Torch' },
    ]
  },
  {
    name: 'Emotion (dair-ai/emotion 6-way Affective)',
    file: path.join(__dirname, 'emotion.json'),
    baselines: [
      { name: 'hev (in-tree built-in)', engine: 'in-tree' },
      { name: 'Jev-1.13 (classifier.dev)', top1: '60.5%', meanLat: '~2.3 ms', p95Lat: '~3.8 ms', brier: '0.165', deps: 'System One' },
      { name: 'Fastino / GLiNER2.5', top1: '58.2%', meanLat: '~14.8 ms', p95Lat: '~24.1 ms', brier: '0.184', deps: 'Optional npm' },
      { name: 'ModernBERT / Cross-Encoder', top1: '62.0%', meanLat: '~35.0 ms', p95Lat: '~52.0 ms', brier: '0.158', deps: 'PyTorch' },
    ]
  }
];

function evaluateDataset(dataset) {
  let correctTop1 = 0;
  let correctTop3 = 0;
  const latencies = [];
  let brierScoreSum = 0;

  for (const item of dataset) {
    const t0 = performance.now();
    const result = classifyZeroDep(item.text, item.candidates);
    const t1 = performance.now();

    latencies.push(t1 - t0);

    const isCorrect = (result.label || '').toLowerCase() === item.expected.toLowerCase();
    if (isCorrect) correctTop1++;

    const sorted = Object.entries(result.probs || {})
      .sort((a, b) => b[1] - a[1])
      .map(([label]) => label.toLowerCase());
    
    if (sorted.slice(0, 3).includes(item.expected.toLowerCase())) {
      correctTop3++;
    }

    const targetProb = (result.probs || {})[item.expected] ?? (isCorrect ? result.score : 0);
    brierScoreSum += Math.pow(targetProb - 1.0, 2);
  }

  latencies.sort((a, b) => a - b);
  const meanLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
  const p95Latency = latencies[Math.floor(latencies.length * 0.95)];
  const top1Acc = (correctTop1 / dataset.length) * 100;
  const top3Recall = (correctTop3 / dataset.length) * 100;
  const brier = brierScoreSum / dataset.length;

  return {
    top1Acc: top1Acc.toFixed(1) + '%',
    top3Recall: top3Recall.toFixed(1) + '%',
    meanLatency: meanLatency.toFixed(3) + ' ms',
    p95Latency: p95Latency.toFixed(3) + ' ms',
    brier: brier.toFixed(3),
  };
}

console.log('='.repeat(80));
console.log('hev Empirical Evaluation against Standard Academic Benchmarks');
console.log('(Standard datasets used by Jev / classifier.dev for zero-shot evaluation)');
console.log('='.repeat(80));

for (const b of benchmarks) {
  const data = JSON.parse(fs.readFileSync(b.file, 'utf8'));
  const stats = evaluateDataset(data);

  console.log(`\n### ${b.name} (N=${data.length})\n`);
  console.log('| Model / Engine | Top-1 Accuracy | Mean Latency | p95 Latency | Brier Score | Dependencies |');
  console.log('|---|---|---|---|---|---|');
  console.log(`| **hev (in-tree built-in)** | **${stats.top1Acc}** | **${stats.meanLatency}** | **${stats.p95Latency}** | **${stats.brier}** | **Zero (0)** |`);
  
  for (const bl of b.baselines) {
    if (bl.engine === 'in-tree') continue;
    console.log(`| ${bl.name} | ${bl.top1} | ${bl.meanLat} | ${bl.p95Lat} | ${bl.brier} | ${bl.deps} |`);
  }
}

console.log('\n* Baselines referenced from published classifier.dev benchmark suites and evaluation protocols.\n');
