export type ComputeDevice = 'cuda' | 'mps' | 'cpu' | 'webgpu' | 'wasm' | 'cloud';

export interface ClassificationMeta {
  label: string;
  score: number;
  probs: Record<string, number>;
  engine: 'gliner' | 'typesafe' | 'builtin' | 'webml' | 'cascade';
  device?: ComputeDevice;
  latencyMs?: number;
}

export interface PredicateMeta {
  value: boolean;
  score: number;
  engine: 'gliner' | 'typesafe' | 'builtin' | 'webml' | 'cascade';
  device?: ComputeDevice;
  latencyMs?: number;
}

export interface HevOptions {
  meta?: boolean;
  apiKey?: string;
  endpoint?: string;
  device?: 'auto' | ComputeDevice;
  engine?: 'webml' | 'builtin' | 'typesafe' | 'gliner' | any;
  model?: string;
  threshold?: number;
}

export type HandlerFn<T = any> = (text: string, meta: ClassificationMeta) => T | Promise<T>;

export type BranchMap<T = any> = Record<string, HandlerFn<T> | T>;

export interface HevFunction {
  // Pattern matching: hev(text, { bug: fn, _: fallback })
  <T = any>(text: string, branches: BranchMap<T>, options?: HevOptions): Promise<T>;

  // Zero-shot classification: hev(text, ['bug', 'feature'])
  (text: string, labels: string[], options?: HevOptions & { meta?: false }): Promise<string>;
  (text: string, labels: string[], options: HevOptions & { meta: true }): Promise<ClassificationMeta>;

  // Boolean predicate: hev(text, 'is spam')
  (text: string, condition: string, options?: HevOptions & { meta?: false }): Promise<boolean>;
  (text: string, condition: string, options: HevOptions & { meta: true }): Promise<PredicateMeta>;

  // Curried versions:
  <T = any>(branches: BranchMap<T>, options?: HevOptions): (text: string, opts?: HevOptions) => Promise<T>;
  (labels: string[], options?: HevOptions): (text: string, opts?: HevOptions) => Promise<string>;

  // Helpers
  detailed(text: string, labels: string[], options?: HevOptions): Promise<ClassificationMeta>;
  score(text: string, conditionOrLabels: string | string[], options?: HevOptions): Promise<number>;
  is(condition: string, options?: HevOptions): (text: string, opts?: HevOptions) => Promise<boolean>;
  pick(labels: string[], options?: HevOptions): (text: string, opts?: HevOptions) => Promise<string>;
  match<T = any>(branches: BranchMap<T>, options?: HevOptions): (text: string, opts?: HevOptions) => Promise<T>;
  device(): ComputeDevice;
}

export class Engine {
  constructor(options?: HevOptions);
  readonly activeDevice: ComputeDevice;
  classify(input: string, labels: string[], options?: HevOptions): Promise<ClassificationMeta>;
  predicate(input: string, condition: string, options?: HevOptions): Promise<PredicateMeta>;
}

export declare const defaultEngine: Engine;
export declare const webmlEngine: Engine;
export declare function detectDevice(): ComputeDevice;

declare const jevish: HevFunction;
declare const hev: HevFunction;
export default jevish;
export { jevish, hev };

