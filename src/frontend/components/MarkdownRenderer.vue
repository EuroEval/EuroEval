<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch, nextTick } from "vue";
import { useRoute, useRouter } from "vue-router";
import type { NavSection } from "@/nav";
import { pageSlug as pageSlugFn } from "@/nav";
import { renderMarkdown, wireTabSync } from "@/markdown";

const props = defineProps<{
  path: string;
  section: NavSection;
  pageSlug?: string;
}>();

const router = useRouter();
const route = useRoute();
const rendered = computed(() => renderMarkdown(props.path));

// Open ancestor <details> blocks and scroll the URL fragment into view —
// triggered after the markdown body is mounted so deep links from the
// search dropdown (e.g. `/faq#how-…`) work even when the target heading
// sits inside a collapsed module.
const scrollToHash = async () => {
  await nextTick();
  const hash = (route.hash || window.location.hash || "").replace(/^#/, "");
  if (!hash) return;
  const el = document.getElementById(hash);
  if (!el) return;
  let cur: HTMLElement | null = el;
  while (cur) {
    if (cur instanceof HTMLDetailsElement && !cur.open) cur.open = true;
    cur = cur.parentElement;
  }
  requestAnimationFrame(() => {
    el.scrollIntoView({ block: "start" });
  });
};

const pageTitle = computed(() => {
  if (!props.pageSlug) return undefined;
  const match = props.section.pages?.find(
    (p) => pageSlugFn(p.path) === props.pageSlug,
  );
  return match?.title;
});

const bodyEl = ref<HTMLDivElement | null>(null);
let tabCleanup: (() => void) | null = null;

// Intercept clicks on internal links inside the rendered markdown so they go
// through vue-router (no full page reload). External links and same-page
// anchors fall through to the browser.
const onClick = (e: MouseEvent) => {
  if (e.defaultPrevented) return;
  if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
  const anchor = (e.target as HTMLElement | null)?.closest("a");
  if (!anchor) return;
  if (anchor.target && anchor.target !== "" && anchor.target !== "_self") return;
  const href = anchor.getAttribute("href");
  if (!href) return;
  // Same-page anchor — let the browser handle it.
  if (href.startsWith("#")) return;
  // Resolve against the current location so we can compare hosts.
  let url: URL;
  try {
    url = new URL(href, window.location.href);
  } catch {
    return;
  }
  if (url.origin !== window.location.origin) return;
  e.preventDefault();
  router.push(url.pathname + url.search + url.hash);
};

watch(
  [bodyEl, rendered],
  async () => {
    if (tabCleanup) {
      tabCleanup();
      tabCleanup = null;
    }
    await nextTick();
    if (bodyEl.value) tabCleanup = wireTabSync(bodyEl.value);
    scrollToHash();
  },
  { immediate: true },
);

watch(() => route.fullPath, scrollToHash);

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
    <div
      v-else
      ref="bodyEl"
      class="markdown-body"
      v-html="rendered.html"
      @click="onClick"
    />
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
