<script setup lang="ts">
import { LANGUAGE_GROUPS, type QueueEntry } from "@/services/github";

function displayGroups(groups: string[]): string {
  if (groups.length === LANGUAGE_GROUPS.length) return "All language groups";
  return groups.join(", ");
}

function statusClass(status: string): string {
  return status.toLowerCase().replace(/\s+/g, "-");
}

defineProps<{ entries: QueueEntry[]; loading: boolean; error: string | null }>();
const emit = defineEmits<{
  refresh: [];
  subscribe: [QueueEntry];
}>();
</script>

<template>
  <section class="queue">
    <div class="queue-header">
      <button
        type="button"
        class="refresh"
        :disabled="loading"
        @click="emit('refresh')"
      >
        {{ loading ? "Refreshing…" : "↻ Refresh" }}
      </button>
    </div>

    <p v-if="error" class="msg error">{{ error }}</p>

    <table v-if="entries.length > 0" class="qtable">
      <thead>
        <tr>
          <th>Model</th>
          <th>Language groups</th>
          <th>Status</th>
          <th class="sub-col"></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="e in entries" :key="e.number">
          <td>
            <a
              :href="`https://huggingface.co/${e.modelId}`"
              target="_blank"
              rel="noopener"
            >
              {{ e.modelId }}
            </a>
          </td>
          <td class="lg">
            <span v-if="e.languageGroups.length === 0" class="muted">—</span>
            <span v-else>{{ displayGroups(e.languageGroups) }}</span>
          </td>
          <td>
            <span
              :class="['status', statusClass(e.status)]"
              :title="
                e.erroredOnVersion
                  ? `Errored on EuroEval v${e.erroredOnVersion}`
                  : undefined
              "
            >
              {{ e.status }}
            </span>
          </td>
          <td class="sub-col">
            <button
              type="button"
              class="subscribe"
              @click="emit('subscribe', e)"
            >
              Subscribe
            </button>
          </td>
        </tr>
      </tbody>
    </table>

    <p v-else-if="!loading && !error" class="empty">
      🎉 No evaluations in the queue right now.
    </p>
  </section>
</template>

<style scoped>
.queue {
  margin-top: 0.5rem;
}

.queue-header {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  margin-bottom: 0.5rem;
}

.refresh {
  padding: 0.35rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: 0.375rem;
  background: var(--color-bg, white);
  font: inherit;
  cursor: pointer;
}

.refresh:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.qtable {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}

.qtable th,
.qtable td {
  border-bottom: 1px solid var(--color-border);
  padding: 0.55rem 0.5rem;
  text-align: left;
  vertical-align: top;
}

.qtable th {
  font-weight: 600;
}

.lg {
  color: var(--color-text-muted, #555);
}

.muted {
  color: var(--color-text-muted, #888);
}

.status {
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
}

.status.evaluating {
  background: rgba(40, 167, 69, 0.12);
  color: #1a7f37;
}

.status.waiting {
  background: rgba(108, 117, 125, 0.18);
  color: #57606a;
}

.status.error {
  background: rgba(220, 53, 69, 0.14);
  color: #b00020;
}

.status.waiting-for-bug-fix {
  background: rgba(255, 165, 0, 0.18);
  color: #b15400;
}

.sub-col {
  width: 1%;
  white-space: nowrap;
}

.subscribe {
  padding: 0.3rem 0.7rem;
  border: 1px solid var(--color-border);
  border-radius: 0.375rem;
  background: var(--color-bg, white);
  font: inherit;
  cursor: pointer;
}

.empty {
  padding: 1.5rem 0;
  color: var(--color-text-muted, #777);
  text-align: center;
}

.msg.error {
  margin: 0.75rem 0;
  padding: 0.5rem 0.75rem;
  border-radius: 0.375rem;
  background: rgba(220, 53, 69, 0.1);
  color: var(--color-danger, #b00020);
}
</style>
