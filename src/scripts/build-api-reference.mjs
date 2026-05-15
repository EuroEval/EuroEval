// Vite plugin that keeps `src/frontend/md/api.md` in sync with the EuroEval
// Python package source. It shells out to `build_api_reference.py`, which uses
// `ast` to walk the package and emit one big markdown file the docs site
// renders like any other page.
//
// Lifecycle:
//   * Once at plugin load — so the file exists before Vite scans the markdown
//     glob.
//   * On `buildStart` — to be safe for production builds.
//   * On any `.py` change under `src/euroeval/` during `vite dev` — followed
//     by a full reload so the in-browser markdown picks up the new content.

import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "../..");
const SCRIPT = path.join(REPO_ROOT, "src/scripts/build_api_reference.py");
const SOURCE_DIR = path.join(REPO_ROOT, "src/euroeval");
const OUTPUT = path.join(REPO_ROOT, "src/frontend/md/api.md");

function regenerate() {
  const result = spawnSync("python3", [SCRIPT], {
    stdio: ["ignore", "inherit", "inherit"],
    cwd: REPO_ROOT,
  });
  if (result.status !== 0) {
    console.error(
      `[api-reference] generator failed with exit code ${result.status}`,
    );
  }
}

// Run once at plugin load so api.md is guaranteed to exist before Vite reads
// the markdown glob during module-graph initialisation.
regenerate();

export default function apiReferencePlugin() {
  return {
    name: "euroeval:api-reference",

    buildStart() {
      regenerate();
    },

    configureServer(server) {
      // Vite's chokidar instance watches `src/`, so .py changes already
      // bubble through. We listen for them and regenerate before triggering
      // a full reload so the page reflects the new source.
      const onChange = (file) => {
        if (!file.startsWith(SOURCE_DIR)) return;
        if (!file.endsWith(".py")) return;
        regenerate();
        // Invalidate the cached markdown module so `import.meta.glob` re-reads
        // it, then force the browser to reload.
        const mod = server.moduleGraph.getModuleById(OUTPUT);
        if (mod) server.moduleGraph.invalidateModule(mod);
        server.ws.send({ type: "full-reload" });
      };
      server.watcher.on("change", onChange);
      server.watcher.on("add", onChange);
      server.watcher.on("unlink", onChange);
    },
  };
}
