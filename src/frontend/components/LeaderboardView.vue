<script setup lang="ts">
import { computed, ref, watch } from "vue";
import LeaderboardTable from "@/components/LeaderboardTable.vue";
import LeaderboardScatter from "@/components/LeaderboardScatter.vue";
import { loadLeaderboard, type LeaderboardTable as LBTable } from "@/leaderboard";

const props = defineProps<{
  stem: string;
  title: string;
}>();

type TabId = "all" | "nlu" | "all-scatter" | "nlu-scatter";

const tab = ref<TabId>("all");

const allTable = ref<LBTable | null>(null);
const nluTable = ref<LBTable | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

const loadFor = async (stem: string) => {
  loading.value = true;
  error.value = null;
  allTable.value = null;
  nluTable.value = null;
  try {
    const [a, n] = await Promise.all([
      loadLeaderboard(`${stem}_all`),
      loadLeaderboard(`${stem}_nlu`),
    ]);
    allTable.value = a ?? null;
    nluTable.value = n ?? null;
    if (!a && !n) {
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
    tab.value = "all";
    loadFor(s);
  },
  { immediate: true },
);

const activeTable = computed<LBTable | null>(() => {
  if (tab.value === "all" || tab.value === "all-scatter") return allTable.value;
  return nluTable.value;
});

const tabs: { id: TabId; label: string }[] = [
  { id: "all", label: "Generative Leaderboard" },
  { id: "nlu", label: "NLU Leaderboard" },
  { id: "all-scatter", label: "Generative Scatter Plot" },
  { id: "nlu-scatter", label: "NLU Scatter Plot" },
];
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
    </nav>

    <div v-if="loading" class="lb-status">Loading leaderboard…</div>
    <div v-else-if="error" class="lb-status error">{{ error }}</div>
    <template v-else>
      <template v-if="tab === 'all' || tab === 'nlu'">
        <LeaderboardTable v-if="activeTable" :table="activeTable" />
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

.lb-status {
  padding: 1.5rem 0;
  color: var(--color-muted);
}

.lb-status.error {
  color: var(--color-danger);
}
</style>
