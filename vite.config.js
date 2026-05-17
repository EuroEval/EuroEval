import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import fs from "fs";
import path from "path";
import seoFiles from "./src/scripts/build-seo-files.mjs";
import apiReference from "./src/scripts/build-api-reference.mjs";

function readPackageVersion() {
  const pyproject = fs.readFileSync(
    path.resolve(__dirname, "pyproject.toml"),
    "utf-8",
  );
  const match = pyproject.match(/^version\s*=\s*"([^"]+)"/m);
  if (!match) throw new Error("Could not find version in pyproject.toml");
  return match[1].replace(/\.dev\d*$/, "");
}

export default defineConfig({
  plugins: [vue(), apiReference(), seoFiles()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src/frontend"),
    },
  },
  define: {
    __PACKAGE_VERSION__: JSON.stringify(readPackageVersion()),
  },
  server: {
    proxy: {
      "/api": {
        target:
          process.env.VERCEL_DEPLOYMENT_URL || "https://euroeval.vercel.app",
        changeOrigin: true,
        secure: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            return "vendor";
          }
        },
      },
    },
    chunkSizeWarningLimit: 1000,
  },
});
