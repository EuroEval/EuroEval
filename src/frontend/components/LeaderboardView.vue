<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from "vue";
import LeaderboardTable from "@/components/LeaderboardTable.vue";
import LeaderboardScatter from "@/components/LeaderboardScatter.vue";
import {
  loadLeaderboard,
  loadLeaderboardCsv,
  loadLeaderboardMetadata,
  type LeaderboardTable as LBTable,
  type LeaderboardMetadata,
} from "@/leaderboard";

const props = defineProps<{
  stem: string;
  title: string;
}>();

type CategoryId = "generative" | "instruct" | "all_models";
type ViewId = "table" | "scatter";

const activeCategory = ref<CategoryId>("generative");
const activeView = ref<ViewId>("table");

const generativeTable = ref<LBTable | null>(null);
const instructTable = ref<LBTable | null>(null);
const allModelsTable = ref<LBTable | null>(null);
const generativeMetadata = ref<LeaderboardMetadata | null>(null);
const instructMetadata = ref<LeaderboardMetadata | null>(null);
const allModelsMetadata = ref<LeaderboardMetadata | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

const loadFor = async (stem: string) => {
  loading.value = true;
  error.value = null;
  generativeTable.value = null;
  instructTable.value = null;
  allModelsTable.value = null;
  generativeMetadata.value = null;
  instructMetadata.value = null;
  allModelsMetadata.value = null;
  try {
    const [g, c, a, gMeta, cMeta, aMeta] = await Promise.all([
      loadLeaderboard(`${stem}_generative`),
      loadLeaderboard(`${stem}_instruct`),
      loadLeaderboard(`${stem}_all_models`),
      loadLeaderboardMetadata(`${stem}_generative`),
      loadLeaderboardMetadata(`${stem}_instruct`),
      loadLeaderboardMetadata(`${stem}_all_models`),
    ]);
    generativeTable.value = g ?? null;
    instructTable.value = c ?? null;
    allModelsTable.value = a ?? null;
    generativeMetadata.value = gMeta ?? null;
    instructMetadata.value = cMeta ?? null;
    allModelsMetadata.value = aMeta ?? null;
    if (!g && !c && !a) {
      error.value = `Leaderboard for ${stem.charAt(0).toUpperCase() + stem.slice(1)} is on the way!`
    }
  } catch (e) {
    error.value = (e as Error).message;
  } finally {
    loading.value = false;
  }
};

watch(
  () => props.stem,
  (s) => {
    // Reset the category on language switch, but keep the user's table/scatter
    // preference - switching language shouldn't kick you out of scatter view.
    activeCategory.value = "instruct";
    loadFor(s);
  },
  { immediate: true },
);

const activeTable = computed<LBTable | null>(() => {
  if (activeCategory.value === "generative") return generativeTable.value;
  if (activeCategory.value === "instruct") return instructTable.value;
  return allModelsTable.value;
});

const activeMetadata = computed<LeaderboardMetadata | null>(() => {
  if (activeCategory.value === "generative") return generativeMetadata.value;
  if (activeCategory.value === "instruct") return instructMetadata.value;
  return allModelsMetadata.value;
});

const lastUpdated = computed<string | null>(() => {
  const notes = activeMetadata.value?.annotate?.notes;
  if (!notes) return null;
  const match = notes.match(/Last updated: (.+)$/);
  if (!match) return null;

  const timestampStr = match[1];
  // Parse "2026-06-20 16:42:37 CET" format (strip timezone for simplicity)
  const dateMatch = timestampStr.match(/(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})/);
  if (!dateMatch) return null;

  const then = new Date(dateMatch[1]).getTime();
  const now = Date.now();
  const diffMs = now - then;

  if (diffMs < 0) return timestampStr; // Future date, show as-is

  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);
  const diffWeeks = Math.floor(diffDays / 7);
  const diffMonths = Math.floor(diffDays / 30);

  if (diffSecs < 60) return `${diffSecs} second${diffSecs !== 1 ? 's' : ''} ago`;
  if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  if (diffWeeks < 4) return `${diffWeeks} week${diffWeeks !== 1 ? 's' : ''} ago`;
  return `${diffMonths} month${diffMonths !== 1 ? 's' : ''} ago`;
});

const MULTILINGUAL_STEMS = new Set([
  "european",
  "baltic",
  "finnic",
  "germanic",
  "romance",
  "scandinavian",
  "slavic",
  "west_germanic",
]);
const isMultilingual = computed(() => MULTILINGUAL_STEMS.has(props.stem));

const categoryTabs: { id: CategoryId; label: string }[] = [
  { id: "instruct", label: "Chat" },
  { id: "generative", label: "Generative" },
  { id: "all_models", label: "All Models" },
];

const viewTabs: { id: ViewId; label: string }[] = [
  { id: "table", label: "Leaderboard" },
  { id: "scatter", label: "Scatter Plot" },
];

// Sliding-pill indicators for the two tab groups. Each indicator is a
// separate absolutely-positioned element measured off the active button's
// own layout, so it works regardless of label width - a plain CSS
// transition can't do this on its own since the buttons aren't fixed-width.
const categoryTabRefs = ref<HTMLButtonElement[]>([]);
const viewTabRefs = ref<HTMLButtonElement[]>([]);
const categoryIndicator = ref({ left: "0px", width: "0px" });
const viewIndicator = ref({ left: "0px", width: "0px" });

const measureIndicator = (
  buttons: HTMLButtonElement[],
  activeIndex: number,
  target: { left: string; width: string },
) => {
  const el = buttons[activeIndex];
  if (!el) return;
  target.left = `${el.offsetLeft}px`;
  target.width = `${el.offsetWidth}px`;
};

const syncIndicators = async () => {
  await nextTick();
  measureIndicator(
    categoryTabRefs.value,
    categoryTabs.findIndex((t) => t.id === activeCategory.value),
    categoryIndicator.value,
  );
  measureIndicator(
    viewTabRefs.value,
    viewTabs.findIndex((t) => t.id === activeView.value),
    viewIndicator.value,
  );
};

watch([activeCategory, activeView], syncIndicators);
onMounted(() => {
  syncIndicators();
  window.addEventListener("resize", syncIndicators);
});
onUnmounted(() => {
  window.removeEventListener("resize", syncIndicators);
});

// Which CSV stem the current category corresponds to, for the download button.
const activeStem = computed<string>(() => {
  if (activeCategory.value === "generative") return `${props.stem}_generative`;
  if (activeCategory.value === "instruct") return `${props.stem}_instruct`;
  return `${props.stem}_all_models`;
});

const downloading = ref(false);

// Embed dialog.
const embedOpen = ref(false);
const embedCopied = ref(false);

const embedUrl = computed(() => {
  const base =
    typeof window !== "undefined"
      ? `${window.location.origin}${window.location.pathname}`
      : `https://euroeval.com/leaderboards/${props.stem}`;
  return `${base}?embed=1`;
});

const embedSnippet = computed(
  () =>
    `<iframe src="${embedUrl.value}" width="100%" height="640" frameborder="0" style="border: 1px solid #d0d7de; border-radius: 6px;" loading="lazy" referrerpolicy="no-referrer" title="EuroEval leaderboard"></iframe>`,
);

const copyEmbed = async () => {
  try {
    await navigator.clipboard.writeText(embedSnippet.value);
    embedCopied.value = true;
    setTimeout(() => {
      embedCopied.value = false;
    }, 1600);
  } catch {
    /* clipboard unavailable; user can copy manually */
  }
};

const downloadCsv = async () => {
  if (downloading.value) return;
  downloading.value = true;
  try {
    const text = await loadLeaderboardCsv(activeStem.value);
    if (!text) return;
    const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeStem.value}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } finally {
    downloading.value = false;
  }
};
</script>

<template>
  <div class="lb-view">
    <h1 class="lb-title" v-html="title" />
    <p class="lb-help">
      See the
      <router-link to="/leaderboards">leaderboard page</router-link>
      for more information about all the columns.
    </p>
    <p class="lb-help">
      Want a model evaluated? Submit it on the
      <router-link to="/evaluation-queue">Evaluation Queue</router-link>.
    </p>

    <aside class="lb-note" role="note">
      <span class="lb-note-icon" aria-hidden="true">ℹ️</span>
      <p>
        Scores use fixed <strong>subsets</strong> — click a dataset for splits; hover a score for failure rates (<span class="fail-flag">*</span> = ≥10% failed).
        <router-link to="/faq#why-do-some-model-names-end-in-val"><code>(val)</code></router-link> = validation split.
      </p>
    </aside>

    <nav class="lb-tabs" role="tablist">
      <span class="lb-tab-indicator" :style="categoryIndicator" />
      <button
        v-for="t in categoryTabs"
        ref="categoryTabRefs"
        :key="t.id"
        type="button"
        role="tab"
        :aria-selected="activeCategory === t.id"
        :class="['lb-tab', { active: activeCategory === t.id }]"
        @click="activeCategory = t.id"
      >
        {{ t.label }}
      </button>
    </nav>

    <div class="lb-view-toggle" role="tablist">
      <span class="lb-view-indicator" :style="viewIndicator" />
      <button
        v-for="v in viewTabs"
        ref="viewTabRefs"
        :key="v.id"
        type="button"
        role="tab"
        :aria-selected="activeView === v.id"
        :class="['lb-view-option', { active: activeView === v.id }]"
        @click="activeView = v.id"
      >
        <svg
          v-if="v.id === 'table'"
          class="lb-view-icon"
          viewBox="0 0 16 16"
          aria-hidden="true"
        >
          <rect x="1" y="6" width="3" height="8" rx="0.5" fill="currentColor" />
          <rect x="6.5" y="3" width="3" height="11" rx="0.5" fill="currentColor" />
          <rect x="12" y="8" width="3" height="6" rx="0.5" fill="currentColor" />
        </svg>
        <svg v-else class="lb-view-icon" viewBox="0 0 16 16" aria-hidden="true">
          <path
            d="M2 2v11a1 1 0 0 0 1 1h11"
            stroke="currentColor"
            stroke-width="1.3"
            fill="none"
            stroke-linecap="round"
          />
          <circle cx="5.5" cy="9" r="1.1" fill="currentColor" />
          <circle cx="9" cy="5.5" r="1.1" fill="currentColor" />
          <circle cx="12" cy="8" r="1.1" fill="currentColor" />
          <circle cx="7" cy="11.5" r="1.1" fill="currentColor" />
        </svg>
        {{ v.label }}
      </button>
    </div>

    <div v-if="embedOpen" class="embed-modal" @click.self="embedOpen = false">
      <div class="embed-dialog" role="dialog" aria-labelledby="embed-title">
        <div class="embed-header">
          <h2 id="embed-title">Embed this leaderboard</h2>
          <button
            class="embed-close"
            type="button"
            aria-label="Close"
            @click="embedOpen = false"
          >
            ×
          </button>
        </div>
        <p>
          Paste this snippet into any HTML page to embed the live
          leaderboard. The iframe stays in sync with the published data.
        </p>
        <textarea class="embed-snippet" readonly :value="embedSnippet" />
        <div class="embed-actions">
          <button class="embed-copy" type="button" @click="copyEmbed">
            {{ embedCopied ? "Copied!" : "Copy snippet" }}
          </button>
          <a class="embed-preview" :href="embedUrl" target="_blank" rel="noopener">
            Preview embed →
          </a>
        </div>
      </div>
    </div>

    <div v-if="loading" class="lb-status">Loading leaderboard…</div>
    <div v-else-if="error" class="lb-status error">{{ error }}</div>
    <template v-else>
      <template v-if="activeView === 'table'">
        <LeaderboardTable
          v-if="activeTable"
          :table="activeTable"
          :heatmap-score-cols="isMultilingual"
          :leaderboard-name="title"
          :last-updated="lastUpdated"
        >
          <template #actions>
            <button
              class="lb-download"
              type="button"
              :disabled="downloading"
              :title="`Download ${activeStem}.csv`"
              @click="downloadCsv"
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M8 1.5a.75.75 0 0 1 .75.75v6.69l2.22-2.22a.75.75 0 1 1 1.06 1.06l-3.5 3.5a.75.75 0 0 1-1.06 0l-3.5-3.5a.75.75 0 1 1 1.06-1.06l2.22 2.22V2.25A.75.75 0 0 1 8 1.5zM2.75 12a.75.75 0 0 1 .75.75v1.25c0 .14.11.25.25.25h8.5c.14 0 .25-.11.25-.25v-1.25a.75.75 0 1 1 1.5 0v1.25c0 .97-.78 1.75-1.75 1.75h-8.5A1.75 1.75 0 0 1 2 14V12.75A.75.75 0 0 1 2.75 12z"
                />
              </svg>
              {{ downloading ? "Downloading…" : "Download CSV" }}
            </button>
            <button
              class="lb-embed"
              type="button"
              title="Embed this leaderboard on another site"
              @click="embedOpen = true"
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M4.78 5.22a.75.75 0 0 1 0 1.06L3.06 8l1.72 1.72a.75.75 0 1 1-1.06 1.06L1.47 8.53a.75.75 0 0 1 0-1.06L3.72 5.22a.75.75 0 0 1 1.06 0zm6.44 0a.75.75 0 0 1 1.06 0l2.25 2.25a.75.75 0 0 1 0 1.06l-2.25 2.25a.75.75 0 1 1-1.06-1.06L12.94 8l-1.72-1.72a.75.75 0 0 1 0-1.06zM9.55 2.04a.75.75 0 0 1 .41.98l-3 8a.75.75 0 1 1-1.4-.53l3-8a.75.75 0 0 1 .99-.45z"
                />
              </svg>
              Embed
            </button>
          </template>         </LeaderboardTable>
         <div v-else class="lb-status">
           This leaderboard variant has no data.
         </div>
      </template>       <template v-else>
         <LeaderboardScatter
           v-if="activeTable"
           :table="activeTable"
         />
       </template>
    </template>
  </div>
</template>

<style scoped>
.lb-view {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.lb-title {
  font-size: 2rem;
  font-weight: 300;
  margin: 0 0 0.25rem;
  padding-bottom: 0.3em;
  border-bottom: 1px solid var(--color-border);
}

.lb-title :deep(a) {
  color: inherit;
  text-decoration: none;
}

.lb-help {
  color: var(--color-muted);
  font-size: 0.9rem;
  margin: 0;
}

.lb-note {
  display: flex;
  gap: 0.5rem;
  align-items: baseline;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-left: 3px solid var(--color-link);
  border-radius: 6px;
  padding: 0.45rem 0.7rem;
  margin-top: 0.5rem;
}

.lb-note-icon {
  flex: none;
}

.lb-note p {
  margin: 0;
  font-size: 0.82rem;
  color: var(--color-muted);
  line-height: 1.4;
}

.lb-note strong {
  color: var(--color-text);
}

.lb-note .fail-flag {
  color: var(--color-danger, #b00020);
  font-size: 1.15em;
  font-weight: 700;
}

.lb-note code {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: 3px;
  padding: 0 0.25rem;
  font-size: 0.8rem;
}

.lb-tabs {
  position: relative;
  display: inline-flex;
  gap: 0.2rem;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 999px;
  padding: 0.25rem;
  margin-top: 0.5rem;
  width: fit-content;
  max-width: 100%;
  overflow-x: auto;
  scrollbar-width: none;
}

.lb-tab-indicator {
  position: absolute;
  top: 0.25rem;
  bottom: 0.25rem;
  left: 0;
  border-radius: 999px;
  background: var(--color-link);
  filter: brightness(0.85);
  transition:
    left 0.25s ease,
    width 0.25s ease;
}

.lb-tab {
  position: relative;
  z-index: 1;
  background: transparent;
  border: 0;
  border-radius: 999px;
  color: var(--color-muted);
  padding: 0.4rem 0.85rem;
  cursor: pointer;
  font: inherit;
  font-size: 0.85rem;
  font-weight: 500;
  white-space: nowrap;
  transition: color 0.2s ease;
}

.lb-tab:hover {
  color: var(--color-text);
}

.lb-tab.active {
  color: #fff;
}

/* Deliberately quieter than .lb-tabs above - the category is the primary
   choice, this is just "how to look at it". Plain underline tabs, not
   pills, so it doesn't compete visually with the category selector. */
.lb-view-toggle {
  position: relative;
  display: inline-flex;
  gap: 1rem;
  border-bottom: 1px solid var(--color-border);
  margin-top: 0.7rem;
}

.lb-view-indicator {
  position: absolute;
  bottom: -1px;
  left: 0;
  height: 2px;
  background: var(--color-muted);
  transition:
    left 0.25s ease,
    width 0.25s ease;
}

.lb-view-option {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  background: transparent;
  border: 0;
  color: var(--color-muted);
  padding: 0.35rem 0.1rem 0.5rem;
  cursor: pointer;
  font: inherit;
  font-size: 0.78rem;
  transition: color 0.2s ease;
}

.lb-view-icon {
  width: 13px;
  height: 13px;
  flex: none;
}

.lb-view-option:hover {
  color: var(--color-text);
}

.lb-view-option.active {
  color: var(--color-text);
}

.lb-download {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.3rem 0.65rem;
  font-size: 0.8rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  align-self: center;
}

.lb-download svg {
  width: 13px;
  height: 13px;
}

.lb-download:hover {
  border-color: var(--color-link);
  color: var(--color-link);
}

.lb-download:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.lb-embed {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.3rem 0.65rem;
  font-size: 0.8rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  align-self: center;
}

.lb-embed svg {
  width: 13px;
  height: 13px;
}

.lb-embed:hover {
  border-color: var(--color-link);
  color: var(--color-link);
}

.embed-modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
  padding: 1rem;
}

.embed-dialog {
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 1.2rem 1.4rem 1.4rem;
  max-width: 560px;
  width: 100%;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}

.embed-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.embed-header h2 {
  font-size: 1.05rem;
  margin: 0;
  font-weight: 500;
}

.embed-close {
  background: transparent;
  border: 0;
  color: var(--color-muted);
  font-size: 1.4rem;
  line-height: 1;
  cursor: pointer;
  padding: 0 0.4rem;
}

.embed-close:hover {
  color: var(--color-text);
}

.embed-dialog p {
  color: var(--color-muted);
  font-size: 0.85rem;
  margin: 0 0 0.75rem;
}

.embed-snippet {
  width: 100%;
  min-height: 110px;
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.6rem 0.75rem;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.78rem;
  resize: vertical;
}

.embed-actions {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 0.75rem;
}

.embed-copy {
  background: var(--color-link);
  color: #fff;
  border: 0;
  border-radius: 4px;
  padding: 0.45rem 0.9rem;
  font-size: 0.85rem;
  cursor: pointer;
}

.embed-copy:hover {
  background: var(--color-link-hover);
}

.embed-preview {
  color: var(--color-link);
  font-size: 0.85rem;
  text-decoration: none;
}

.embed-preview:hover {
  text-decoration: underline;
}

.lb-status {
  padding: 1.5rem 0;
  color: var(--color-muted);
}

.lb-status.error {
  color: var(--color-danger);
}
</style>
