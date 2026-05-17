<script setup lang="ts">
import { onMounted, ref } from "vue";
import { listEvaluators, type EvaluatorCount } from "@/services/github";

const TOP_N = 10;
const CACHE_KEY = "euroeval:hall-of-fame:evaluators";
const CACHE_TTL_MS = 60 * 60 * 1000;

interface CachedPayload {
  ts: number;
  data: EvaluatorCount[];
}

const entries = ref<EvaluatorCount[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);

function readCache(): EvaluatorCount[] | null {
  try {
    const raw = sessionStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedPayload;
    if (Date.now() - parsed.ts > CACHE_TTL_MS) return null;
    return parsed.data;
  } catch {
    return null;
  }
}

function writeCache(data: EvaluatorCount[]): void {
  try {
    sessionStorage.setItem(
      CACHE_KEY,
      JSON.stringify({ ts: Date.now(), data } satisfies CachedPayload),
    );
  } catch {
    // ignore quota errors
  }
}

function medal(index: number): string {
  return ["🥇", "🥈", "🥉"][index] ?? "";
}

onMounted(async () => {
  const cached = readCache();
  if (cached) {
    entries.value = cached.slice(0, TOP_N);
    return;
  }
  loading.value = true;
  try {
    const all = await listEvaluators();
    writeCache(all);
    entries.value = all.slice(0, TOP_N);
  } catch (e) {
    error.value = (e as Error).message;
  } finally {
    loading.value = false;
  }
});
</script>

<template>
  <aside class="hof">
    <h2>🏆 Contributor Hall of Fame</h2>
    <p class="sub">Most models evaluated</p>

    <p v-if="loading" class="msg">Loading…</p>
    <p v-else-if="error" class="msg err">{{ error }}</p>
    <ol v-else-if="entries.length > 0" class="list">
      <li v-for="(e, i) in entries" :key="e.login">
        <span class="rank">{{ medal(i) || `${i + 1}.` }}</span>
        <a
          class="user"
          :href="`https://github.com/${e.login}`"
          target="_blank"
          rel="noopener"
        >
          <img :src="e.avatarUrl" :alt="e.login" loading="lazy" />
          <span class="login">@{{ e.login }}</span>
        </a>
        <span class="count">{{ e.count }}</span>
      </li>
    </ol>
    <p v-else class="msg">No submissions yet.</p>
  </aside>
</template>

<style scoped>
.hof {
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1rem 1rem 0.75rem;
  background: var(--color-bg, white);
}

h2 {
  margin: 0;
  font-size: 1.05rem;
}

.sub {
  margin: 0.15rem 0 0.75rem;
  color: var(--color-text-muted, #777);
  font-size: 0.8rem;
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.list li {
  display: grid;
  grid-template-columns: 1.5rem 1fr auto;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.88rem;
}

.rank {
  text-align: center;
  color: var(--color-text-muted, #888);
  font-variant-numeric: tabular-nums;
}

.user {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  text-decoration: none;
  color: inherit;
  min-width: 0;
}

.user:hover .login {
  text-decoration: underline;
}

.user img {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  flex-shrink: 0;
}

.login {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.count {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: var(--color-text-muted, #555);
}

.msg {
  margin: 0;
  font-size: 0.85rem;
  color: var(--color-text-muted, #777);
}

.msg.err {
  color: var(--color-danger, #b00020);
}
</style>
