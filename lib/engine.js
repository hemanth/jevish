import { classifyZeroDep, judgePredicateZeroDep } from './zero-dep.js';

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

/**
 * Unified execution engine supporting 3 tiers:
 * 1. Fastino / GLiNER2.5 (local neural, hardware accelerated: CUDA → MPS → CPU)
 * 2. TypeSafe System One (cloud, if TYPESAFE_API_KEY present)
 * 3. Zero-dep in-tree engine (default, always available, sub-0.1ms, CPU cache, browser-ready)
 */
export class Engine {
  constructor(options = {}) {
    this.apiKey = options.apiKey !== undefined ? options.apiKey : (typeof process !== 'undefined' ? process.env?.TYPESAFE_API_KEY : undefined);
    this.endpoint = options.endpoint || 'https://api.typesafe.ai/v1/systemone';
    this.device = options.device || 'auto';
    this.cascade = options.cascade ?? false;
    this.cascadeThreshold = options.cascadeThreshold ?? 2.0;
  }

  get activeDevice() {
    return this.device === 'auto' ? detectDevice() : this.device;
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

    // Speculative Cascade Mode: In-Tree Fast Path (<0.05ms) + Cloud Escalation
    if (isCascade) {
      const local = classifyZeroDep(input, labels);
      const isConfident = (local.margin >= threshold && local.score >= 0.70);

      if (isConfident || !this.apiKey) {
        return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
      }

      try {
        const cloud = await this._callJevChoice(input, labels);
        if (cloud) {
          return { ...cloud, engine: 'cascade', fastPath: false, device: 'cloud' };
        }
      } catch {
        // Fall back to local on error
      }
      return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
    }

    // Tier 1: Local Neural GLiNER2.5 if installed
    const gliner = await loadGliner();
    if (gliner && typeof gliner.classify === 'function') {
      try {
        const result = await gliner.classify(input, labels);
        return { ...result, engine: 'gliner', device: this.activeDevice };
      } catch {
        // Fall through on error
      }
    }

    // Tier 2: Cloud System One if API key is provided
    if (this.apiKey) {
      try {
        const cloud = await this._callJevChoice(input, labels);
        if (cloud) return cloud;
      } catch {
        // Fall through to in-tree on network/API failure
      }
    }

    // Tier 3: Zero-dependency built-in engine (always reliable, 0ms, offline)
    const result = classifyZeroDep(input, labels);
    return { ...result, engine: 'builtin', device: 'cpu' };
  }

  async predicate(input, condition, options = {}) {
    const isCascade = options.cascade ?? this.cascade;

    if (isCascade) {
      const isQuestion = condition.trim().endsWith('?');
      const local = judgePredicateZeroDep(input, condition);
      const isConfident = !isQuestion && (local.evidence >= 8.0);

      if (isConfident || !this.apiKey) {
        return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
      }

      try {
        const cloud = await this._callJevPredicate(input, condition);
        if (cloud) {
          return { ...cloud, engine: 'cascade', fastPath: false, device: 'cloud' };
        }
      } catch {
        // Fall back
      }
      return { ...local, engine: 'cascade', fastPath: true, device: 'cpu' };
    }

    if (this.apiKey) {
      try {
        const cloud = await this._callJevPredicate(input, condition);
        if (cloud) return cloud;
      } catch {
        // Fall through
      }
    }

    const result = judgePredicateZeroDep(input, condition);
    return { ...result, engine: 'builtin', device: 'cpu' };
  }
}

export const defaultEngine = new Engine();
