import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

function resolve(dir) {
  return path.resolve(__dirname, dir);
}

// https://vitejs.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    ignored: [
      resolve("src-tauri/binaries/**"),
      resolve("src-tauri/target/**"),
      resolve(
        "src-tauri/target/debug/binaries/build/aarch64-apple-darwin/debug/install/lib/torch/utils/model_dump/skeleton.html"
      ),
    ],
  },
  watch: {
    ignored: [
      resolve("src-tauri/binaries/**"),
      resolve("src-tauri/target/**"),
      resolve(
        "src-tauri/target/debug/binaries/build/aarch64-apple-darwin/debug/install/lib/torch/utils/model_dump/skeleton.html"
      ),
    ],
  },
}));
