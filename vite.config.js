import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import path from "path";
import seoFiles from "./src/scripts/build-seo-files.mjs";

export default defineConfig({
  plugins: [vue(), seoFiles()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src/frontend"),
    },
  },
});
