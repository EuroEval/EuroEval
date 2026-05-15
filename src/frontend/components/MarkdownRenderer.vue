<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch, nextTick } from "vue";
import type { NavSection } from "@/nav";
import { pageSlug as pageSlugFn } from "@/nav";
import { renderMarkdown, wireTabSync } from "@/markdown";

const props = defineProps<{
  path: string;
  section: NavSection;
  pageSlug?: string;
}>();

const rendered = computed(() => renderMarkdown(props.path));

const pageTitle = computed(() => {
  if (!props.pageSlug) return undefined;
  const match = props.section.pages?.find(
    (p) => pageSlugFn(p.path) === props.pageSlug,
  );
  return match?.title;
});

const bodyEl = ref<HTMLDivElement | null>(null);
let tabCleanup: (() => void) | null = null;

watch(
  [bodyEl, rendered],
  async () => {
    if (tabCleanup) {
      tabCleanup();
      tabCleanup = null;
    }
    await nextTick();
    if (bodyEl.value) tabCleanup = wireTabSync(bodyEl.value);
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  if (tabCleanup) tabCleanup();
});
</script>

<template>
  <article class="md-page">
    <nav class="breadcrumb" v-if="pageTitle">
      <router-link :to="`/${section.id}`">{{ section.title }}</router-link>
      <span class="sep">›</span>
      <span class="current" v-html="pageTitle" />
    </nav>
    <div v-if="!rendered" class="error">Failed to load content: {{ path }}</div>
    <div v-else ref="bodyEl" class="markdown-body" v-html="rendered.html" />
  </article>
</template>

<style scoped>
.md-page {
  max-width: 100%;
}

.breadcrumb {
  font-size: 0.85rem;
  color: var(--color-muted);
  margin-bottom: 1rem;
}

.breadcrumb a {
  color: var(--color-muted);
  text-decoration: none;
}

.breadcrumb a:hover {
  color: var(--color-link);
}

.breadcrumb .sep {
  margin: 0 0.4rem;
}

.error {
  color: var(--color-danger);
}
</style>
