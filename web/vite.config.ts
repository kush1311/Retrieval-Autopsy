import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The demo deploy is a static build with no backend: `VITE_DEMO_ONLY=1` makes the app
// replay pre-recorded traces from src/demo/traces. Anyone evaluating this will click
// a link, not clone the repo, and a cold-start demo that demands an API key gets
// closed immediately.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // No rewrite. The API serves its routes *under* /api, so stripping the prefix
      // here would forward /api/meta to /meta and 404. The rewrite was correct when
      // the routers were mounted at the root; it became a bug the moment they moved,
      // and the failure is invisible from the client — a 404 body, not a config error.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/stream": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
