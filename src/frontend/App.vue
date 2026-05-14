<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import TopHeader from "@/components/TopHeader.vue";
import SectionTabs from "@/components/SectionTabs.vue";
import SideNav from "@/components/SideNav.vue";
import MarkdownRenderer from "@/components/MarkdownRenderer.vue";
import TableOfContents from "@/components/TableOfContents.vue";
import {
  defaultSection,
  findSection,
  resolveMdPath,
} from "@/nav";

const route = useRoute();

const sectionId = computed(() => {
  const param = route.params.section as string | undefined;
  return param || defaultSection.id;
});

const pageSlug = computed(
  () => (route.params.page as string | undefined) || undefined,
);

const activeSection = computed(
  () => findSection(sectionId.value) || defaultSection,
);

const mdPath = computed(() =>
  resolveMdPath(sectionId.value, pageSlug.value),
);

const hasSidebar = computed(
  () => Boolean(activeSection.value.pages && activeSection.value.pages.length),
);

const hasToc = computed(() => Boolean(activeSection.value.toc));

const layoutClass = computed(() => {
  if (hasSidebar.value) return "with-sidebar";
  if (hasToc.value) return "with-toc";
  return "with-none";
});
</script>

<template>
  <div class="app">
    <TopHeader />
    <SectionTabs :active-id="activeSection.id" />
    <div class="page" :class="layoutClass">
      <SideNav v-if="hasSidebar" :section="activeSection" :active-page-slug="pageSlug" />
      <main class="content-wrap">
        <MarkdownRenderer
          v-if="mdPath"
          :path="mdPath"
          :section="activeSection"
          :page-slug="pageSlug"
        />
        <div v-else class="error">Page not found.</div>
      </main>
      <TableOfContents v-if="hasToc && mdPath" :path="mdPath" />
    </div>
    <footer class="site-footer">
      Made by
      <a href="https://www.saattrupdan.com/" target="_blank" rel="noopener">
        Dan Saattrup Smart
      </a>
      at the
      <a href="https://alexandra.dk/" target="_blank" rel="noopener">
        Alexandra Institute
      </a>
    </footer>
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.page {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr;
  max-width: 1280px;
  width: 100%;
  margin: 0 auto;
  padding: 0 1.5rem;
  gap: 2rem;
}

.page.with-sidebar {
  grid-template-columns: 240px 1fr;
}

.page.with-toc {
  grid-template-columns: 1fr 220px;
}

.content-wrap {
  min-width: 0;
  padding: 1.5rem 0 3rem;
}

.error {
  color: var(--color-danger);
  padding: 2rem 0;
}

.site-footer {
  background: var(--color-footer-bg);
  color: var(--color-footer-text);
  padding: 1rem 1.5rem;
  text-align: center;
  font-size: 0.85rem;
  border-top: 1px solid var(--color-border);
}

.site-footer a {
  color: var(--color-link);
}

@media (max-width: 1024px) {
  .page.with-toc {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .page.with-sidebar {
    grid-template-columns: 1fr;
  }
}
</style>
