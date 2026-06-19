import { createApp } from "vue";
import App from "@/App.vue";
import router from "@/router";
import "@/styles/main.css";

// After a deploy, hashed chunk filenames change and the old ones are removed
// from the server. A browser still running the previous build references the
// dead URLs, so its dynamic imports fail with a "loading dynamically imported
// module" error. Vite fires `vite:preloadError` in that case; reload once to
// pull a fresh index.html with the current chunk hashes. The session flag
// stops a genuinely missing chunk from causing a reload loop, and is cleared
// on any successful leaderboard load (see leaderboard.ts).
window.addEventListener("vite:preloadError", () => {
  const key = "euroeval:stale-chunk-reload";
  try {
    if (sessionStorage.getItem(key)) return;
    sessionStorage.setItem(key, "1");
  } catch {
    return;
  }
  window.location.reload();
});

const app = createApp(App);
app.use(router);
app.mount("#app");
