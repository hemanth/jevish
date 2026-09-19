# jevish

Semantic pattern matching and zero-shot judgment in JavaScript. Sub-millisecond, calibrated, local-first.

[**Live Interactive Playground →**](https://hemanth.github.io/jevish/)

```bash
npm install jevish
```

## Quick start

```js
import jevish from 'jevish';

await jevish('Checkout button returns 500 internal server error', {
  'bug @ >0.8': (t, meta) => fileJira(t, meta.score),
  'bug':        (t) => queueTriage(t),
  'billing':    (t) => openStripe(t),
  _:            (t) => logToInbox(t),
});
```

`jevish()` evaluates semantic pattern handlers, zero-shot arrays, or boolean predicates in a single forward pass. That's the whole API.

## Zero-shot classification

Pass an array of labels to get the winning category:

```js
const category = await jevish('Can you provide an invoice for last month?', [
  'bug',
  'feature',
  'billing',
]);
// => 'billing'
```

Access calibrated probabilities via `jevish.detailed()`:

```js
const meta = await jevish.detailed('Database connection pool exhausted', ['bug', 'feature']);
console.log(meta.score); // 0.96
console.log(meta.probs); // { bug: 0.96, feature: 0.04 }
```

## Functional pipelines & currying

Every mode auto-curries when called with only the patterns:

```js
const triage = jevish(['bug', 'feature', 'billing']);
const isSpam = jevish.is('spam');

const tickets = await fetchInbox();
const categories = await Promise.all(tickets.map(triage));
const spamEmails = await emails.filterAsync?.(isSpam);
```

## Empirical benchmark

Evaluated on standard zero-shot benchmarks used by TypeSafe Jev:

### Hugging Face Benchmarks (N=100 per task)
| Task / Dataset | `jevish (in-tree)` | `jevish (cascade)` | Jev (TypeSafe API) |
|---|---|---|---|
| **Intent Routing** (`banking77`) | **86.0%** (0.05 ms) | **99.0%** (29 ms) | **100.0%** (139 ms) |
| **Spam Guardrails** (`sms_spam`) | **72.0%** (0.02 ms) | **98.0%** (134 ms) | **98.0%** (146 ms) |
| **Topic Triage** (`ag_news`) | **33.0%** (0.06 ms) | **79.0%** (140 ms) | **82.0%** (149 ms) |

Run `npm run bench` to reproduce live across all canonical Hugging Face datasets.

## Runtime & devices

Runs anywhere: Node.js, Bun, Deno, Cloudflare Workers, and modern browsers (8.6 kB). Detects `CUDA` → `MPS` → `CPU` (or `WebGPU` → `WASM` in browser) via `jevish.device()`. [View Execution Path Blueprint →](docs/execution-path.svg)


```bash
npm run demo
npm run bench
npm run playground
```

Runs the multi-mode demonstration, reproduces the benchmark suite, and launches the local interactive playground at `http://localhost:3456`.

## License

MIT © [Hemanth.HM](https://h3manth.com)
