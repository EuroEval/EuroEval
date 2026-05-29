<script setup lang="ts">
import type { NavSection } from "@/nav";
import { groupedPages, pageSlug, pageUrl } from "@/nav";
import { drawerOpen, closeDrawer } from "@/drawer";
import { computed, watch } from "vue";
import { useRoute } from "vue-router";

const props = defineProps<{
  section: NavSection;
  activePageSlug: string | undefined;
}>();

const groups = computed(() => groupedPages(props.section));

const isActivePage = (page: { path?: string; csv?: string; slug?: string }) =>
  pageSlug(page as any) === props.activePageSlug;

const route = useRoute();
watch(
  () => route.fullPath,
  () => {
    closeDrawer();
  },
);
</script>

<template>
  <div
    class="drawer-backdrop"
    :class="{ open: drawerOpen }"
    @click="closeDrawer"
  ></div>
  <aside class="side-nav" :class="{ open: drawerOpen }">
    <router-link
      :to="`/${section.id}`"
      class="side-title"
      :class="{ active: !activePageSlug }"
    >
      {{ section.title }}
    </router-link>
    <div v-for="(group, gi) in groups" :key="gi" class="side-group">
      <div v-if="group.title" class="side-group-title">{{ group.title }}</div>
      <ul class="side-list">
        <li v-for="page in group.pages" :key="page.title">
          <router-link
            :to="pageUrl(section.id, page)"
            class="side-link"
            :class="{ active: isActivePage(page) }"
            v-html="page.title"
          />
        </li>
      </ul>
    </div>
  </aside>
</template>

<style scoped>
.side-nav {
  padding: 1.5rem 0;
  font-size: 0.85rem;
  position: sticky;
  top: 130px;
  align-self: start;
  max-height: calc(100vh - 150px);
  overflow-y: auto;
}

.side-title {
  display: block;
  font-weight: 500;
  color: var(--color-link);
  text-decoration: none;
  margin-bottom: 0.5rem;
}

.side-title:not(.active) {
  color: var(--color-text);
}

.side-group + .side-group {
  margin-top: 1rem;
}

.side-group-title {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-muted);
  margin: 0.4rem 0 0.3rem;
}

.side-list {
  list-style: none;
  margin: 0;
  padding: 0;
}

.side-list li {
  margin: 0;
}

.side-link {
  display: block;
  padding: 0.3rem 0;
  color: var(--color-text);
  text-decoration: none;
}

.side-link:hover {
  color: var(--color-link);
}

.side-link.active {
  color: var(--color-link);
  font-weight: 500;
}

.drawer-backdrop {
  display: none;
}

@media (max-width: 768px) {
  .side-nav {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 260px;
    background: var(--color-bg);
    border-right: 1px solid var(--color-border);
    padding: 1.25rem 1rem;
    transform: translateX(-100%);
    transition: transform 0.2s ease;
    z-index: 30;
    overflow-y: auto;
    max-height: none;
  }
  .side-nav.open {
    transform: translateX(0);
  }
  .drawer-backdrop {
    display: block;
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
    z-index: 25;
  }
  .drawer-backdrop.open {
    opacity: 1;
    pointer-events: auto;
  }
}
</style>
