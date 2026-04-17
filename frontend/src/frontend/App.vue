<script setup lang="ts">
import Navigation from "@/components/Navigation.vue";
import MarkdownRenderer from "@/components/MarkdownRenderer.vue";
import { useRoute } from "vue-router";
import { computed } from "vue";

const route = useRoute();
const activePath = computed(() => {
  return route.path.replace(/^\//, "") || "README.md";
});
</script>

<template>
  <div class="layout">
    <Navigation />
    <main class="content">
      <MarkdownRenderer :path="activePath" />
    </main>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100vh;
}

.content {
  flex: 1;
  overflow-y: auto;
  padding: 2rem;
  max-width: 800px;
  margin: 0 auto;
}
</style>
