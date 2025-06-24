/* tslint:disable */
/* eslint-disable */
export function main(): void;
export class TunnelIDCore {
  free(): void;
  /**
   * Initialize with your biometric fuzzy key parameters
   */
  constructor(dim: number, d: number, epsilon: number, sigma: number);
  /**
   * Generate key from biometric vector and tag
   * Returns JS objecr (serialized FSKeyOutput) with {key, sketch, g_a, beta, tag}
   */
  keygen(x: Float64Array, tag: string): any;
  /**
   * Recover key from a new biometric vector and stored public metadata
   * Returns key as bytes or null if recovery fails
   */
  keyrecover(x_prime: Float64Array, sketch_json: string): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_tunnelidcore_free: (a: number, b: number) => void;
  readonly tunnelidcore_new: (a: number, b: number, c: number, d: number) => number;
  readonly tunnelidcore_keygen: (a: number, b: number, c: number, d: number, e: number) => any;
  readonly tunnelidcore_keyrecover: (a: number, b: number, c: number, d: number, e: number) => any;
  readonly main: () => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_3: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
