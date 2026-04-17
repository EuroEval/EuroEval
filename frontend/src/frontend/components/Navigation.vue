<script setup lang="ts">
import { ref, onMounted } from "vue";
import { useRoute } from "vue-router";
import yaml from "js-yaml";

const route = useRoute();
const navConfig = ref<any>(null);

onMounted(async () => {
  const response = await fetch("/config.yaml");
  const text = await response.text();
  navConfig.value = yaml.load(text) as any;
});

const isActive = (path: string) => {
  return route.path === `/docs/${path}` || route.path === `/docs/${path}/`;
};
</script>

<template>
  <nav class="sidebar">
    <div v-if="navConfig">
      <div class="site-title">{{ navConfig.title }}</div>
      <div v-for="section in navConfig.nav" :key="section.title">
        <div class="section-title">{{ section.title }}</div>
        <ul class="section-pages">
          <li v-for="page in section.pages" :key="page.path">
            <a
              :href="`/docs/${page.path}`"
              :class="{ active: isActive(page.path) }"
            >
              {{ page.title || page.path }}
            </a>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>

<style scoped>
.sidebar {
  width: 300px;
  overflow-y: auto;
  border-right: 1px solid #e0e0e0;
  padding: 1rem;
}

.site-title {
  font-size: 1.2rem;
  font-weight: bold;
  margin-bottom: 1rem;
}

.section-title {
  font-weight: bold;
  margin: 1rem 0 0.5rem;
}

.section-pages {
  list-style: none;
  padding: 0;
}

.section-pages li a {
  display: block;
  padding: 0.25rem 0;
  text-decoration: none;
  color: inherit;
}

.section-pages li a.active {
  font-weight: bold;
}
</style>
