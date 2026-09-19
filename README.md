# hev

Semantic pattern matching and zero-shot judgment in JavaScript. Sub-millisecond, calibrated, local-first.

[**Live Interactive Playground →**](https://hemanth.github.io/hev/)

```bash
npm install hev
```

## Quick start

```js
import hev from 'hev';

await hev('Checkout button returns 500 internal server error', {
  'bug @ >0.8': (t, meta) => fileJira(t, meta.score),
  'bug':        (t) => queueTriage(t),
  'billing':    (t) => openStripe(t),
  _:            (t) => logToInbox(t),
});
```

`hev()` evaluates semantic pattern handlers, zero-shot arrays, or boolean predicates in a single forward pass. That's the whole API.

## Zero-shot classification

Pass an array of labels to get the winning category:

```js
const category = await hev('Can you provide an invoice for last month?', [
  'bug',
  'feature',
  'billing',
]);
// => 'billing'
```

Access calibrated probabilities via `hev.detailed()`:

```js
const meta = await hev.detailed('Database connection pool exhausted', ['bug', 'feature']);
console.log(meta.score); // 0.96
console.log(meta.probs); // { bug: 0.96, feature: 0.04 }
```

## Functional pipelines & currying

Every mode auto-curries when called with only the patterns:

```js
const triage = hev(['bug', 'feature', 'billing']);
const isSpam = hev.is('spam');

const tickets = await fetchInbox();
const categories = await Promise.all(tickets.map(triage));
const spamEmails = await emails.filterAsync?.(isSpam);
```

## Empirical benchmark

Evaluated on standard zero-shot benchmarks used by Jev (`classifier.dev`):

### AG News (4-way topic)
| Model / Engine | Top-1 Acc | Latency (mean) | Dependencies |
|---|---|---|---|
| **hev (in-tree)** | **84.0%** | **0.088 ms** | **Zero (0)** |
| Jev-1.13 | 87.7% | ~2.1 ms | System One |
| Fastino / GLiNER2.5 | 81.2% | ~14.2 ms | Optional npm |

### Emotion (6-way affective)
| Model / Engine | Top-1 Acc | Latency (mean) | Dependencies |
|---|---|---|---|
| **hev (in-tree)** | **83.0%** | **0.034 ms** | **Zero (0)** |
| Jev-1.13 | 60.5% | ~2.3 ms | System One |
| Fastino / GLiNER2.5 | 58.2% | ~14.8 ms | Optional npm |

Run `npm run bench` to reproduce across `bench/ag_news.json` and `bench/emotion.json`.

## Runtime & devices

Runs anywhere: Node.js, Bun, Deno, Cloudflare Workers, and modern browsers (8.6 kB). Detects `CUDA` → `MPS` → `CPU` (or `WebGPU` → `WASM` in browser) via `hev.device()`.


```bash
npm run demo
npm run bench
npm run playground
```

Runs the multi-mode demonstration, reproduces the benchmark suite, and launches the local interactive playground at `http://localhost:3456`.

## License

MIT © [Hemanth.HM](https://h3manth.com)
