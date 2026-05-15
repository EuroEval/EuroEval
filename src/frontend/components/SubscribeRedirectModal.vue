<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref } from "vue";

const props = defineProps<{ url: string }>();
const emit = defineEmits<{ close: [] }>();

const countdown = ref(3);
let timer: ReturnType<typeof setInterval> | null = null;
let opened = false;

function openNow() {
  if (opened) return;
  opened = true;
  window.open(props.url, "_blank", "noopener");
  emit("close");
}

onMounted(() => {
  timer = setInterval(() => {
    countdown.value -= 1;
    if (countdown.value <= 0) {
      if (timer) clearInterval(timer);
      openNow();
    }
  }, 1000);
});

onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
});
</script>

<template>
  <div class="modal-backdrop" @click.self="emit('close')">
    <div class="modal" role="dialog" aria-live="polite">
      <h3>Taking you to the GitHub issue</h3>
      <p>
        You can subscribe to updates on this evaluation directly on its GitHub
        issue. Opening it in a new tab in
        <strong>{{ countdown }}</strong>
        second<span v-if="countdown !== 1">s</span>…
      </p>
      <div class="actions">
        <button type="button" class="secondary" @click="emit('close')">
          Cancel
        </button>
        <button type="button" class="primary" @click="openNow">
          Open now
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal {
  background: var(--color-bg, white);
  color: inherit;
  padding: 1.5rem 1.75rem;
  border-radius: 0.5rem;
  max-width: 28rem;
  width: calc(100% - 2rem);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}

h3 {
  margin: 0 0 0.5rem;
}

p {
  margin: 0 0 1rem;
}

.actions {
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
}

button {
  padding: 0.45rem 0.95rem;
  border-radius: 0.375rem;
  font: inherit;
  cursor: pointer;
  border: 1px solid var(--color-border);
}

button.primary {
  background: var(--color-link, #1f6feb);
  color: white;
  border-color: transparent;
}

button.secondary {
  background: var(--color-bg, white);
}
</style>
