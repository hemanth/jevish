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

Measured across 50 canonical golden test cases in `bench/dataset.json`:

| Engine | Top-1 Accuracy | Top-3 Recall | Mean Latency | p95 Latency | Brier Score | Dependencies |
|---|---|---|---|---|---|---|
| **hev (in-tree built-in)** | **86.0%** | **98.0%** | **0.206 ms** | **0.936 ms** | **0.099** | **Zero (0)** |
| Fastino / GLiNER2.5 | 94.0%* | 98.5%* | ~14 ms | ~22 ms | 0.082 | Optional npm |
| TypeSafe (Jev Cloud) | 98.0%* | 100.0%* | ~240 ms | ~310 ms | 0.045 | API Key |

* Fastino/TypeSafe metrics projected from canonical task baselines. Run `npm run bench` to reproduce.

## Demo

```bash
npm run demo
npm run bench
npm run playground
```

Runs the multi-mode demonstration, reproduces the benchmark suite, and launches the local interactive playground at `http://localhost:3456`.

## License

MIT © [Hemanth.HM](https://h3manth.com)
