<script setup lang="ts">
import { navConfig, sectionUrl } from "@/nav";

defineProps<{ activeId: string }>();
</script>

<template>
  <nav class="tabs">
    <div class="tabs-inner">
      <router-link
        v-for="section in navConfig.sections"
        :key="section.id"
        :to="sectionUrl(section)"
        class="tab"
        :class="{ active: section.id === activeId }"
      >
        {{ section.title }}
      </router-link>
    </div>
  </nav>
</template>

<style scoped>
.tabs {
  background: var(--color-header-bg);
  color: var(--color-header-text);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.18);
  position: sticky;
  top: 64px;
  z-index: 9;
}

.tabs-inner {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 2rem;
  display: flex;
  gap: 1.5rem;
  overflow-x: auto;
  scrollbar-width: none;
}

.tabs-inner::-webkit-scrollbar {
  display: none;
}

.tab {
  color: rgba(255, 255, 255, 0.75);
  text-decoration: none;
  padding: 0.8rem 0.1rem 0.7rem;
  font-size: 0.85rem;
  white-space: nowrap;
  border-bottom: 2px solid transparent;
}

.tab:hover {
  color: #fff;
}

.tab.active {
  color: #fff;
  border-bottom-color: #fff;
  font-weight: 500;
}
</style>
