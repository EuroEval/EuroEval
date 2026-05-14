<script setup lang="ts">
import { computed } from "vue";
import type { NavSection } from "@/nav";
import { pageSlug as pageSlugFn } from "@/nav";
import { renderMarkdown } from "@/markdown";

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
</script>

<template>
  <article class="md-page">
    <nav class="breadcrumb" v-if="pageTitle">
      <router-link :to="`/${section.id}`">{{ section.title }}</router-link>
      <span class="sep">›</span>
      <span class="current">{{ pageTitle }}</span>
    </nav>
    <div v-if="!rendered" class="error">Failed to load content: {{ path }}</div>
    <div v-else class="markdown-body" v-html="rendered.html" />
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
