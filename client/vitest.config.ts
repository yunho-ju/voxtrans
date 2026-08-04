import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Separate from vite.config.ts deliberately: `vitest` reads
// `vitest.config.*` in preference to `vite.config.*` when both exist, but
// plain `vite` (the `dev`/`build` scripts) only ever reads `vite.config.ts`
// and never looks at this file -- so adding a test runner cannot perturb
// the existing dev/build pipeline.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    // jsdom disables the Storage API (localStorage/sessionStorage) under
    // the default "about:blank" origin per spec; App.tsx persists to
    // localStorage (voxtrans_server_url, voxtrans_input_device), so tests
    // need a real http(s) origin for it to exist at all.
    environmentOptions: {
      jsdom: { url: "http://localhost/" },
    },
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
