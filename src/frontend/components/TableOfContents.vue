<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount, ref, watch, nextTick } from "vue";
import { useRoute } from "vue-router";
import type { TocItem } from "@/markdown";
import { renderMarkdown } from "@/markdown";

const props = defineProps<{
  path: string;
}>();

const route = useRoute();

const toc = computed<TocItem[]>(() => renderMarkdown(props.path)?.toc ?? []);

const activeId = ref<string | null>(null);

let observer: IntersectionObserver | null = null;

const attachObserver = () => {
  observer?.disconnect();
  const ids = toc.value.map((t) => t.id);
  if (ids.length === 0) return;

  observer = new IntersectionObserver(
    (entries) => {
      // Track visible headings; pick the topmost-visible one.
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length > 0) {
        activeId.value = visible[0].target.id;
        return;
      }
      // Otherwise: pick the last heading above the viewport top.
      const above = ids
        .map((id) => document.getElementById(id))
        .filter((el): el is HTMLElement => !!el)
        .filter((el) => el.getBoundingClientRect().top < 120);
      if (above.length > 0) {
        activeId.value = above[above.length - 1].id;
      }
    },
    { rootMargin: "-80px 0px -70% 0px", threshold: [0, 1] },
  );

  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  }
};

const reattach = async () => {
  await nextTick();
  attachObserver();
};

onMounted(reattach);
onBeforeUnmount(() => observer?.disconnect());
watch(() => props.path, reattach);
watch(() => route.fullPath, reattach);

const onClick = (e: MouseEvent, id: string) => {
  const el = document.getElementById(id);
  if (!el) return;
  e.preventDefault();
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  history.replaceState(null, "", `#${id}`);
};
</script>

<template>
  <aside class="toc" v-if="toc.length">
    <div class="toc-title">Table of contents</div>
    <ul class="toc-list">
      <li
        v-for="item in toc"
        :key="item.id"
        :class="['toc-item', `level-${item.level}`, { active: item.id === activeId }]"
      >
        <a :href="`#${item.id}`" @click="onClick($event, item.id)">
          {{ item.text }}
        </a>
      </li>
    </ul>
  </aside>
</template>

<style scoped>
.toc {
  font-size: 0.85rem;
  padding: 1.5rem 0;
  position: sticky;
  top: 130px;
  align-self: start;
  max-height: calc(100vh - 150px);
  overflow-y: auto;
}

.toc-title {
  font-weight: 600;
  color: var(--color-text);
  margin-bottom: 0.6rem;
}

.toc-list {
  list-style: none;
  margin: 0;
  padding: 0;
  border-left: 1px solid var(--color-border);
}

.toc-item a {
  display: block;
  padding: 0.3rem 0.75rem;
  color: var(--color-muted);
  text-decoration: none;
  border-left: 2px solid transparent;
  margin-left: -1px;
  line-height: 1.35;
}

.toc-item.level-3 a {
  padding-left: 1.5rem;
  font-size: 0.8rem;
}

.toc-item a:hover {
  color: var(--color-link);
}

.toc-item.active a {
  color: var(--color-link);
  border-left-color: var(--color-link);
}

@media (max-width: 1024px) {
  .toc {
    display: none;
  }
}
</style>
