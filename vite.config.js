import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import path from "path";
import seoFiles from "./src/scripts/build-seo-files.mjs";
import apiReference from "./src/scripts/build-api-reference.mjs";

export default defineConfig({
  plugins: [vue(), apiReference(), seoFiles()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src/frontend"),
    },
  },
});
