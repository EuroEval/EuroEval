<script setup lang="ts">
import { computed, ref, watch } from "vue";
import LeaderboardTable from "@/components/LeaderboardTable.vue";
import LeaderboardScatter from "@/components/LeaderboardScatter.vue";
import {
  loadLeaderboard,
  loadLeaderboardCsv,
  type LeaderboardTable as LBTable,
} from "@/leaderboard";

const props = defineProps<{
  stem: string;
  title: string;
}>();

type TabId =
  | "generative"
  | "all_models"
  | "generative-scatter"
  | "all_models-scatter";

const tab = ref<TabId>("generative");

const generativeTable = ref<LBTable | null>(null);
const allModelsTable = ref<LBTable | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

const loadFor = async (stem: string) => {
  loading.value = true;
  error.value = null;
  generativeTable.value = null;
  allModelsTable.value = null;
  try {
    const [g, a] = await Promise.all([
      loadLeaderboard(`${stem}_generative`),
      loadLeaderboard(`${stem}_all_models`),
    ]);
    generativeTable.value = g ?? null;
    allModelsTable.value = a ?? null;
    if (!g && !a) {
      error.value = `No leaderboard data found for "${stem}".`;
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
    tab.value = "generative";
    loadFor(s);
  },
  { immediate: true },
);

const activeTable = computed<LBTable | null>(() => {
  if (tab.value === "generative" || tab.value === "generative-scatter")
    return generativeTable.value;
  return allModelsTable.value;
});

const MULTILINGUAL_STEMS = new Set([
  "european",
  "baltic",
  "finnic",
  "germanic",
  "mainland_scandinavian",
  "romance",
  "slavic",
]);
const isMultilingual = computed(() => MULTILINGUAL_STEMS.has(props.stem));

const tabs: { id: TabId; label: string }[] = [
  { id: "generative", label: "Generative Leaderboard" },
  { id: "generative-scatter", label: "Generative Scatter Plot" },
  { id: "all_models", label: "NLU Leaderboard" },
  { id: "all_models-scatter", label: "NLU Scatter Plot" },
];

// Which CSV stem the current tab corresponds to, for the download button.
const activeStem = computed<string>(() => {
  const suffix =
    tab.value === "generative" || tab.value === "generative-scatter"
      ? "generative"
      : "all_models";
  return `${props.stem}_${suffix}`;
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

    <nav class="lb-tabs" role="tablist">
      <button
        v-for="t in tabs"
        :key="t.id"
        type="button"
        role="tab"
        :aria-selected="tab === t.id"
        :class="['lb-tab', { active: tab === t.id }]"
        @click="tab = t.id"
      >
        {{ t.label }}
      </button>
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
    </nav>

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
      <template v-if="tab === 'generative' || tab === 'all_models'">
        <LeaderboardTable
          v-if="activeTable"
          :table="activeTable"
          :heatmap-score-cols="isMultilingual"
        />
        <div v-else class="lb-status">
          This leaderboard variant has no data.
        </div>
      </template>
      <template v-else>
        <LeaderboardScatter
          v-if="activeTable"
          :table="activeTable"
        />
        <div v-else class="lb-status">
          This leaderboard variant has no data.
        </div>
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

.lb-tabs {
  display: flex;
  gap: 0.5rem;
  border-bottom: 1px solid var(--color-border);
  margin-top: 0.5rem;
  overflow-x: auto;
  scrollbar-width: none;
}

.lb-tab {
  background: transparent;
  border: 0;
  border-bottom: 2px solid transparent;
  color: var(--color-muted);
  padding: 0.6rem 0.75rem;
  cursor: pointer;
  font: inherit;
  font-size: 0.85rem;
  white-space: nowrap;
}

.lb-tab:hover {
  color: var(--color-text);
}

.lb-tab.active {
  color: var(--color-link);
  border-bottom-color: var(--color-link);
  font-weight: 500;
}

.lb-download {
  margin-left: auto;
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
  margin-left: 0.4rem;
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
