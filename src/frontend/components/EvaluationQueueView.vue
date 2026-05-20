<script setup lang="ts">
import { onMounted, ref } from "vue";
import HallOfFame from "@/components/HallOfFame.vue";
import ModelSubmitForm from "@/components/ModelSubmitForm.vue";
import QueueTable from "@/components/QueueTable.vue";
import SubscribeRedirectModal from "@/components/SubscribeRedirectModal.vue";
import { listOpenEvalIssues, type QueueEntry } from "@/services/github";

const entries = ref<QueueEntry[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const subscribeUrl = ref<string | null>(null);
const pending = ref<Map<number, QueueEntry>>(new Map());

async function refresh(opts: { fresh?: boolean } = {}) {
  loading.value = true;
  error.value = null;
  try {
    const fresh = await listOpenEvalIssues({ fresh: opts.fresh });
    const seen = new Set(fresh.map((e) => e.number));
    for (const n of [...pending.value.keys()]) {
      if (seen.has(n)) pending.value.delete(n);
    }
    entries.value = [...fresh, ...pending.value.values()];
  } catch (e) {
    error.value = (e as Error).message;
  } finally {
    loading.value = false;
  }
}

function onSubmitted(payload: {
  url: string;
  modelId: string;
  number: number;
  languageGroups: string[];
}) {
  const entry: QueueEntry = {
    number: payload.number,
    url: payload.url,
    modelId: payload.modelId,
    languageGroups: payload.languageGroups,
    status: "Waiting",
    evaluator: null,
    createdAt: new Date().toISOString(),
  };
  pending.value.set(payload.number, entry);
  if (!entries.value.some((e) => e.number === payload.number)) {
    entries.value = [...entries.value, entry];
  }
  void refresh({ fresh: true });
}

function onSubscribe(entry: QueueEntry) {
  subscribeUrl.value = entry.url;
}

onMounted(refresh);
</script>

<template>
  <div class="eq-view">
    <div class="main">
      <h1>Evaluation Queue</h1>
      <p class="intro">
        Anyone can submit a public model on the Hugging Face Hub for evaluation
        on EuroEval. Submissions become GitHub issues, and our compute servers
        pick them up automatically; results are then merged into the
        leaderboards.
      </p>

      <ModelSubmitForm @submitted="onSubmitted" />
    </div>

    <div class="sidebar">
      <HallOfFame />
    </div>

    <div class="queue-wrap">
      <QueueTable
        :entries="entries"
        :loading="loading"
        :error="error"
        @refresh="refresh({ fresh: true })"
        @subscribe="onSubscribe"
      />
    </div>

    <SubscribeRedirectModal
      v-if="subscribeUrl"
      :url="subscribeUrl"
      @close="subscribeUrl = null"
    />
  </div>
</template>

<style scoped>
.eq-view {
  max-width: 1200px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 260px;
  grid-template-areas:
    "main sidebar"
    "queue sidebar";
  column-gap: 2rem;
  align-items: start;
}

.main {
  grid-area: main;
  min-width: 0;
}

.queue-wrap {
  grid-area: queue;
  min-width: 0;
}

.sidebar {
  grid-area: sidebar;
  position: sticky;
  top: 8rem;
  max-height: calc(100vh - 9rem);
  overflow-y: auto;
}

h1 {
  font-size: 2rem;
  font-weight: 300;
  margin: 0 0 0.25rem;
  padding-bottom: 0.3em;
  border-bottom: 1px solid var(--color-border);
}

.intro {
  margin: 0 0 1.5rem;
  color: var(--color-text-muted, #555);
}

@media (max-width: 900px) {
  .eq-view {
    grid-template-columns: 1fr;
    grid-template-areas:
      "main"
      "sidebar"
      "queue";
    row-gap: 1.5rem;
  }

  .sidebar {
    position: static;
    max-height: none;
    overflow-y: visible;
  }
}
</style>
