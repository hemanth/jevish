import { classifyZeroDep, judgePredicateZeroDep, SEMANTIC_CLUSTERS } from './zero-dep.js';

let glinerModule = null;
let glinerLoadAttempted = false;

/**
 * Probe optional hardware / neural model package dynamically
 */
async function loadGliner() {
  if (glinerLoadAttempted) return glinerModule;
  glinerLoadAttempted = true;
  try {
    // Dynamic import for optional Fastino / GLiNER2.5 package
    glinerModule = await import('@fastino/gliner25-long-context');
  } catch {
    glinerModule = null;
  }
  return glinerModule;
}

/**
 * Detect runtime compute environment and preferred hardware device:
 * - Native (Node/Bun/Deno): CUDA (NVIDIA) → MPS (Apple Silicon Metal) → CPU
 * - Browser: WebGPU (navigator.gpu) → WASM (SIMD) → CPU
 */
export function detectDevice() {
  // Browser context
  if (typeof window !== 'undefined' || (typeof navigator !== 'undefined' && typeof process === 'undefined')) {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      return 'webgpu';
    }
    if (typeof WebAssembly !== 'undefined') {
      return 'wasm';
    }
    return 'cpu';
  }

  // Node.js / Server runtime
  if (typeof process !== 'undefined') {
    if (process.env?.HEV_DEVICE) return process.env.HEV_DEVICE;

    // Check for CUDA
    if (process.env?.CUDA_VISIBLE_DEVICES && process.env.CUDA_VISIBLE_DEVICES !== '-1') {
      return 'cuda';
    }

    // Check for Apple Silicon MPS (Metal Performance Shaders)
    if (process.platform === 'darwin' && (process.arch === 'arm64' || process.env?.PROCESSOR_ARCHITECTURE === 'ARM64')) {
      return 'mps';
    }

    return 'cpu';
  }

  return 'cpu';
}

let webmlKitModule = null;
let webmlLoadAttempted = false;

async function loadWebMLKit() {
  if (webmlLoadAttempted) return webmlKitModule;
  webmlLoadAttempted = true;
  try {
    webmlKitModule = await import('webml-kit');
  } catch {
    webmlKitModule = null;
  }
  return webmlKitModule;
}

/**
 * Unified execution engine supporting 4 tiers:
 * 1. webml-kit / OpenJev Decision Engine (browser WebGPU / WASM, zero-dep fallback)
 * 2. Fastino / GLiNER2.5 (local neural, hardware accelerated: CUDA → MPS → CPU)
 * 3. TypeSafe System One (cloud, if TYPESAFE_API_KEY present)
 * 4. Zero-dep in-tree engine (default, always available, sub-0.1ms, CPU cache, browser-ready)
 */
export class Engine {
  constructor(options = {}) {
    this.type = options.type || options.engine || 'auto';
    this.apiKey = options.apiKey !== undefined ? options.apiKey : (typeof process !== 'undefined' ? process.env?.TYPESAFE_API_KEY : undefined);
    this.endpoint = options.endpoint || 'https://api.typesafe.ai/v1/systemone';
    this.device = options.device || 'auto';
    this.cascade = options.cascade ?? false;
    this.cascadeThreshold = options.cascadeThreshold ?? 2.0;
    this.model = options.model || 'qwen3-0.6b';
    this.clusters = options.clusters || null;
    this.gliner = options.gliner || null;
    this.webmlDecisionEngine = null;
    this.options = options;
  }

  get activeDevice() {
    return this.device === 'auto' ? detectDevice() : this.device;
  }

  async _getWebMLDecisionEngine(options = {}) {
    if (this.webmlDecisionEngine) return this.webmlDecisionEngine;
    const webml = await loadWebMLKit();
    const createFn = webml?.createDecisionEngine || webml?.default?.decision;
    if (typeof createFn === 'function') {
      this.webmlDecisionEngine = createFn({
        model: options.model || this.model,
        mode: options.mode || 'auto',
      });
      await this.webmlDecisionEngine.init();
      return this.webmlDecisionEngine;
    }
    return null;
  }

  async _callJevChoice(input, labels) {
    if (!this.apiKey) return null;
    const res = await fetch(this.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        state: input,
        model: 'jev-latest',
        questions: {
          choice: {
            type: 'choice',
            instructions: 'Select the label that best fits the input text.',
            criteria: labels.reduce((acc, l) => ({ ...acc, [l]: null }), {}),
          },
        },
      }),
    });

    if (res.ok) {
      const data = await res.json();
      const q = data.answers?.choice;
      if (q) {
        return {
          label: q.choice,
          score: q.confidence ?? 1.0,
          probs: q.probabilities ?? {},
          engine: 'typesafe',
          device: 'cloud',
        };
      }
    }
    return null;
  }

  async _callJevPredicate(input, condition) {
    if (!this.apiKey) return null;
    const res = await fetch(this.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        state: input,
        model: 'jev-latest',
        questions: {
          noul: {
            type: 'noul',
            instructions: `Is the statement true about the text: ${condition}`,
          },
        },
      }),
    });

    if (res.ok) {
      const data = await res.json();
      const q = data.answers?.noul;
      if (q) {
        return {
          value: q.noul >= 0.5,
          score: q.noul,
          engine: 'typesafe',
          device: 'cloud',
        };
      }
    }
    return null;
  }

  async classify(input, labels, options = {}) {
    const isCascade = options.cascade ?? this.cascade;
    const threshold = options.cascadeThreshold ?? this.cascadeThreshold;
    const targetEngine = options.engine || options.type || this.type;
    const clusters = options.clusters || this.clusters;

    // Tier 1: webml-kit / OpenJev Decision Engine
    if (targetEngine === 'webml') {
      const decisionEngine = await this._getWebMLDecisionEngine(options);
      if (decisionEngine) {
        const res = await decisionEngine.choice({
          state: input,
          question: 'Select the label that best fits the input text.',
          options: labels,
        });
        return {
          label: res.choice,
          score: res.confidence,
          probs: res.probabilities,
          margin: (res.confidence - 0.5) * 2,
          engine: 'webml',
          device: decisionEngine.isHeuristic ? 'cpu' : 'webgpu',
          latencyMs: res.latencyMs,
        };
      }
    }

    // Speculative Cascade Mode: In-Tree Fast Path (<0.05ms) + Local Neural GLiNER + Cloud Escalation
    if (isCascade) {
      const local = classifyZeroDep(input, labels, { ...options, clusters });
      const isConfident = (local.margin >= threshold && local.score >= 0.70);

      if (isConfident) {
        return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
      }

      // Tier 2: Try local neural GLiNER if available/injected
      const gliner = options.gliner || this.gliner || await loadGliner();
      if (gliner && typeof gliner.classify === 'function') {
        try {
          const neural = await gliner.classify(input, labels, options);
          if (neural) {
            return { ...neural, engine: 'cascade', fastPath: false, device: this.activeDevice };
          }
        } catch {
          // Fall through on error
        }
      }

      // Tier 3: Cloud escalation if API key is provided
      if (this.apiKey) {
        try {
          const cloud = await this._callJevChoice(input, labels);
          if (cloud) {
            return { ...cloud, engine: 'cascade', fastPath: false, device: 'cloud' };
          }
        } catch {
          // Fall back to local on error
        }
      }

      return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
    }

    // Tier 2: Local Neural GLiNER2.5 if installed or provided
    const gliner = options.gliner || this.gliner || await loadGliner();
    if (gliner && typeof gliner.classify === 'function') {
      try {
        const result = await gliner.classify(input, labels, options);
        return { ...result, engine: 'gliner', device: this.activeDevice };
      } catch {
        // Fall through on error
      }
    }

    // Tier 3: Cloud System One if API key is provided
    if (this.apiKey) {
      try {
        const cloud = await this._callJevChoice(input, labels);
        if (cloud) return cloud;
      } catch {
        // Fall through to in-tree on network/API failure
      }
    }

    // Tier 4: Zero-dependency built-in engine (always reliable, 0ms, offline)
    const result = classifyZeroDep(input, labels, { ...options, clusters });
    return { ...result, engine: 'builtin', device: 'cpu' };
  }

  async predicate(input, condition, options = {}) {
    const isCascade = options.cascade ?? this.cascade;
    const targetEngine = options.engine || options.type || this.type;
    const clusters = options.clusters || this.clusters;

    // Tier 1: webml-kit / OpenJev Decision Engine
    if (targetEngine === 'webml') {
      const decisionEngine = await this._getWebMLDecisionEngine(options);
      if (decisionEngine) {
        const res = await decisionEngine.noul({
          state: input,
          statement: condition,
          threshold: options.threshold ?? 0.5,
        });
        return {
          value: res.passed,
          score: res.noul,
          evidence: res.noul * 10,
          engine: 'webml',
          device: decisionEngine.isHeuristic ? 'cpu' : 'webgpu',
          latencyMs: res.latencyMs,
        };
      }
    }

    if (isCascade) {
      const isQuestion = condition.trim().endsWith('?');
      const local = judgePredicateZeroDep(input, condition, { ...options, clusters });
      const isConfident = !isQuestion && (local.evidence >= 8.0);

      if (isConfident) {
        return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
      }

      // Tier 2: Try local neural GLiNER if available/injected
      const gliner = options.gliner || this.gliner || await loadGliner();
      if (gliner && typeof gliner.predicate === 'function') {
        try {
          const neural = await gliner.predicate(input, condition, options);
          if (neural) {
            return { ...neural, engine: 'cascade', fastPath: false, device: this.activeDevice };
          }
        } catch {
          // Fall through
        }
      }

      // Tier 3: Cloud escalation if API key is provided
      if (this.apiKey) {
        try {
          const cloud = await this._callJevPredicate(input, condition);
          if (cloud) {
            return { ...cloud, engine: 'cascade', fastPath: false, device: 'cloud' };
          }
        } catch {
          // Fall back
        }
      }

      return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
    }

    // Tier 2: Local Neural GLiNER2.5 if installed or provided
    const gliner = options.gliner || this.gliner || await loadGliner();
    if (gliner && typeof gliner.predicate === 'function') {
      try {
        const result = await gliner.predicate(input, condition, options);
        return { ...result, engine: 'gliner', device: this.activeDevice };
      } catch {
        // Fall through
      }
    }

    if (this.apiKey) {
      try {
        const cloud = await this._callJevPredicate(input, condition);
        if (cloud) return cloud;
      } catch {
        // Fall through
      }
    }

    const result = judgePredicateZeroDep(input, condition, { ...options, clusters });
    return { ...result, engine: 'builtin', device: 'cpu' };
  }
}

export const defaultEngine = new Engine();
export const webmlEngine = new Engine({ type: 'webml' });
export { SEMANTIC_CLUSTERS };

