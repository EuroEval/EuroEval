<script setup lang="ts">
import type { NavSection } from "@/nav";
import { pageSlug, pageUrl } from "@/nav";
import { drawerOpen, closeDrawer } from "@/drawer";
import { watch } from "vue";
import { useRoute } from "vue-router";

const props = defineProps<{
  section: NavSection;
  activePageSlug: string | undefined;
}>();

const isActive = (path: string) => pageSlug(path) === props.activePageSlug;

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
    <ul class="side-list">
      <li v-for="page in section.pages" :key="page.path">
        <router-link
          :to="pageUrl(section.id, page)"
          class="side-link"
          :class="{ active: isActive(page.path) }"
        >
          {{ page.title }}
        </router-link>
      </li>
    </ul>
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

.side-title.active {
  color: var(--color-link);
}

.side-title:not(.active) {
  color: var(--color-text);
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
