/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" for the static, keyless deploy that replays pre-recorded traces. */
  readonly VITE_DEMO_ONLY?: string;
  readonly VITE_API_BASE?: string;
  readonly VITE_WS_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
