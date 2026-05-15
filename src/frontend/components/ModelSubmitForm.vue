<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { LANGUAGE_GROUPS, submitEvalRequest } from "@/services/github";
import { searchHfModels, type HfModelSuggestion } from "@/services/huggingface";

const emit = defineEmits<{ submitted: [{ url: string; modelId: string }] }>();

const modelId = ref("");
const selectedGroups = ref<Set<string>>(new Set());
const suggestions = ref<HfModelSuggestion[]>([]);
const showSuggestions = ref(false);
const highlight = ref(-1);
const submitting = ref(false);
const errorMsg = ref<string | null>(null);
const successMsg = ref<string | null>(null);

let searchTimer: ReturnType<typeof setTimeout> | null = null;

const canSubmit = computed(
  () =>
    /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(modelId.value.trim()) &&
    selectedGroups.value.size > 0 &&
    !submitting.value,
);

watch(modelId, (v) => {
  if (searchTimer) clearTimeout(searchTimer);
  highlight.value = -1;
  if (!v.trim()) {
    suggestions.value = [];
    showSuggestions.value = false;
    return;
  }
  searchTimer = setTimeout(async () => {
    suggestions.value = await searchHfModels(v.trim(), 15);
    showSuggestions.value = suggestions.value.length > 0;
  }, 180);
});

onBeforeUnmount(() => {
  if (searchTimer) clearTimeout(searchTimer);
});

function toggleGroup(g: string) {
  const next = new Set(selectedGroups.value);
  if (next.has(g)) next.delete(g);
  else next.add(g);
  selectedGroups.value = next;
}

function pickSuggestion(s: HfModelSuggestion) {
  modelId.value = s.id;
  showSuggestions.value = false;
}

function onInputBlur() {
  setTimeout(() => {
    showSuggestions.value = false;
  }, 150);
}

function onInputFocus() {
  if (suggestions.value.length > 0) showSuggestions.value = true;
}

function onKeydown(e: KeyboardEvent) {
  if (!showSuggestions.value || suggestions.value.length === 0) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    highlight.value = (highlight.value + 1) % suggestions.value.length;
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    highlight.value =
      (highlight.value - 1 + suggestions.value.length) % suggestions.value.length;
  } else if (e.key === "Enter" && highlight.value >= 0) {
    e.preventDefault();
    pickSuggestion(suggestions.value[highlight.value]);
  } else if (e.key === "Escape") {
    showSuggestions.value = false;
  }
}

async function onSubmit() {
  if (!canSubmit.value) return;
  submitting.value = true;
  errorMsg.value = null;
  successMsg.value = null;
  const result = await submitEvalRequest(
    modelId.value.trim(),
    Array.from(selectedGroups.value),
  );
  submitting.value = false;
  if (result.ok && result.url) {
    successMsg.value = `Submitted — your model is in the queue.`;
    emit("submitted", { url: result.url, modelId: modelId.value.trim() });
    modelId.value = "";
    selectedGroups.value = new Set();
    suggestions.value = [];
  } else if (result.status === 409 && result.url) {
    errorMsg.value = `This model is already in the queue — see ${result.url}.`;
  } else {
    errorMsg.value = result.error || "Submission failed. Please try again.";
  }
}
</script>

<template>
  <form class="submit-form" @submit.prevent="onSubmit">
    <details class="submit-details">
      <summary class="submit-summary">
        <span class="chev" aria-hidden="true">▸</span>
        <h2>Submit a model</h2>
        <span class="toggle-hint">
          <span class="hint-open">Click to expand</span>
          <span class="hint-close">Click to collapse</span>
        </span>
      </summary>

      <p class="hint">
        Pick a public model on the
        <a href="https://huggingface.co/models" target="_blank" rel="noopener">
          Hugging Face Hub
        </a>
        and choose which language groups it should be evaluated on. API-only
        models are not accepted for automatic evaluation, as it can be quite
        expensive, but we urge you to
        <a
          href="https://github.com/EuroEval/EuroEval/issues/new?template=model_evaluation_request.yaml"
          target="_blank"
          rel="noopener"
        >open an issue</a>
        if you think we missed one.
      </p>

    <label class="field">
      <span class="label">Model ID</span>
      <div class="autocomplete">
        <input
          v-model="modelId"
          type="text"
          placeholder="e.g. meta-llama/Llama-3.1-8B"
          autocomplete="off"
          spellcheck="false"
          @focus="onInputFocus"
          @blur="onInputBlur"
          @keydown="onKeydown"
        />
        <ul
          v-if="showSuggestions"
          class="suggestions"
          role="listbox"
        >
          <li
            v-for="(s, i) in suggestions"
            :key="s.id"
            role="option"
            :aria-selected="i === highlight"
            :class="{ active: i === highlight }"
            @mousedown.prevent="pickSuggestion(s)"
          >
            <span class="sid">{{ s.id }}</span>
            <span v-if="s.downloads" class="dl">
              {{ s.downloads.toLocaleString() }} ↓
            </span>
          </li>
        </ul>
      </div>
    </label>

    <fieldset class="field">
      <legend class="label">Language groups to evaluate the model on</legend>
      <div class="groups">
        <label v-for="g in LANGUAGE_GROUPS" :key="g" class="group">
          <input
            type="checkbox"
            :checked="selectedGroups.has(g)"
            @change="toggleGroup(g)"
          />
          <span>{{ g }}</span>
        </label>
      </div>
    </fieldset>

    <div class="actions">
      <button type="submit" :disabled="!canSubmit">
        {{ submitting ? "Submitting…" : "Submit for evaluation" }}
      </button>
    </div>

      <p v-if="errorMsg" class="msg error">{{ errorMsg }}</p>
      <p v-if="successMsg" class="msg success">{{ successMsg }}</p>
    </details>
  </form>
</template>

<style scoped>
.submit-form {
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  background: var(--color-card-bg, transparent);
  margin-bottom: 2rem;
  overflow: hidden;
}

h2 {
  margin: 0;
  font-size: 1.1rem;
}

.submit-summary {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  list-style: none;
  cursor: pointer;
  user-select: none;
  padding: 0.85rem 1.25rem;
  background: var(--color-hover-bg, rgba(31, 111, 235, 0.06));
  transition: background 0.12s ease;
}

.submit-summary::-webkit-details-marker {
  display: none;
}

.submit-summary:hover,
.submit-summary:focus-visible {
  background: var(--color-hover-bg-strong, rgba(31, 111, 235, 0.12));
  outline: none;
}

.submit-details[open] .submit-summary {
  border-bottom: 1px solid var(--color-border);
}

.chev {
  display: inline-block;
  font-size: 0.85rem;
  transition: transform 0.15s ease;
  color: var(--color-link, #1f6feb);
}

.submit-details[open] .chev {
  transform: rotate(90deg);
}

.toggle-hint {
  margin-left: auto;
  font-size: 0.8rem;
  color: var(--color-text-muted, #666);
}

.hint-open,
.hint-close {
  display: none;
}

.submit-details:not([open]) .hint-open {
  display: inline;
}

.submit-details[open] .hint-close {
  display: inline;
}

.submit-details > :not(.submit-summary) {
  padding-left: 1.5rem;
  padding-right: 1.5rem;
}

.submit-details > .hint {
  padding-top: 1rem;
}

.submit-details > .msg:last-child {
  margin-bottom: 1.25rem;
}

.submit-details > .actions {
  padding-bottom: 1.25rem;
}

.hint {
  margin: 0 0 1rem;
  font-size: 0.9rem;
  color: var(--color-text-muted, #555);
}

.field {
  display: block;
  margin-bottom: 1rem;
  border: 0;
  padding: 0;
}

.label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.35rem;
}

.autocomplete {
  position: relative;
}

.autocomplete input {
  width: 100%;
  padding: 0.5rem 0.65rem;
  border: 1px solid var(--color-border);
  border-radius: 0.375rem;
  font: inherit;
  background: var(--color-bg, white);
  color: inherit;
}

.suggestions {
  position: absolute;
  z-index: 10;
  top: 100%;
  left: 0;
  right: 0;
  margin: 2px 0 0;
  padding: 0;
  list-style: none;
  border: 1px solid var(--color-border);
  border-radius: 0.375rem;
  background: var(--color-bg, white);
  max-height: 18rem;
  overflow-y: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.suggestions li {
  padding: 0.4rem 0.65rem;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: 0.9rem;
}

.suggestions li.active,
.suggestions li:hover {
  background: var(--color-hover-bg, #f0f4f8);
}

.dl {
  color: var(--color-text-muted, #777);
  font-variant-numeric: tabular-nums;
}

.groups {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 0.4rem 1rem;
}

.group {
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
  font-size: 0.9rem;
  cursor: pointer;
}

.actions {
  display: flex;
  justify-content: flex-end;
}

button[type="submit"] {
  padding: 0.55rem 1.1rem;
  border: 0;
  border-radius: 0.375rem;
  background: var(--color-link, #1f6feb);
  color: white;
  font: inherit;
  cursor: pointer;
}

button[type="submit"]:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.msg {
  margin: 0.75rem 0 0;
  padding: 0.5rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.9rem;
}

.msg.error {
  background: rgba(220, 53, 69, 0.1);
  color: var(--color-danger, #b00020);
}

.msg.success {
  background: rgba(40, 167, 69, 0.12);
  color: #1a7f37;
}
</style>
