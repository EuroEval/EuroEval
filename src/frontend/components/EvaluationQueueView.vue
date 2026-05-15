<script setup lang="ts">
import { onMounted, ref } from "vue";
import ModelSubmitForm from "@/components/ModelSubmitForm.vue";
import QueueTable from "@/components/QueueTable.vue";
import SubscribeRedirectModal from "@/components/SubscribeRedirectModal.vue";
import { listOpenEvalIssues, type QueueEntry } from "@/services/github";

const entries = ref<QueueEntry[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const subscribeUrl = ref<string | null>(null);

async function refresh() {
  loading.value = true;
  error.value = null;
  try {
    entries.value = await listOpenEvalIssues();
  } catch (e) {
    error.value = (e as Error).message;
  } finally {
    loading.value = false;
  }
}

function onSubmitted() {
  void refresh();
}

function onSubscribe(entry: QueueEntry) {
  subscribeUrl.value = entry.url;
}

onMounted(refresh);
</script>

<template>
  <div class="eq-view">
    <h1>Evaluation Queue</h1>
    <p class="intro">
      Anyone can submit a public model on the Hugging Face Hub for evaluation
      on EuroEval. Submissions become GitHub issues, and our compute servers
      pick them up automatically; results are then merged into the
      leaderboards. We don't accept API-only models as they can get quite costly to
      evaluate, but you can still
      <a
        href="https://github.com/EuroEval/EuroEval/issues/new?template=model_evaluation_request.yaml"
        target="_blank"
        rel="noopener"
      >
        submit an evaluation request
      </a>
      for it!
    </p>

    <ModelSubmitForm @submitted="onSubmitted" />

    <QueueTable
      :entries="entries"
      :loading="loading"
      :error="error"
      @refresh="refresh"
      @subscribe="onSubscribe"
    />

    <SubscribeRedirectModal
      v-if="subscribeUrl"
      :url="subscribeUrl"
      @close="subscribeUrl = null"
    />
  </div>
</template>

<style scoped>
.eq-view {
  max-width: 920px;
}

h1 {
  margin: 0 0 0.5rem;
}

.intro {
  margin: 0 0 1.5rem;
  color: var(--color-text-muted, #555);
}
</style>
