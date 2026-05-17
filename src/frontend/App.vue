<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import TopHeader from "@/components/TopHeader.vue";
import SectionTabs from "@/components/SectionTabs.vue";
import SideNav from "@/components/SideNav.vue";
import MarkdownRenderer from "@/components/MarkdownRenderer.vue";
import LeaderboardView from "@/components/LeaderboardView.vue";
import EvaluationQueueView from "@/components/EvaluationQueueView.vue";
import TableOfContents from "@/components/TableOfContents.vue";
import {
  defaultSection,
  findPage,
  findSection,
  resolveMdPath,
} from "@/nav";
import { useHead } from "@/useHead";

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

const activePage = computed(() =>
  findPage(activeSection.value, pageSlug.value),
);

const isLeaderboard = computed(() => Boolean(activePage.value?.csv));

const customView = computed(() =>
  !activePage.value ? activeSection.value.view : undefined,
);

const mdPath = computed(() => {
  if (isLeaderboard.value) return undefined;
  if (customView.value) return undefined;
  return resolveMdPath(sectionId.value, pageSlug.value);
});

const hasSidebar = computed(
  () => Boolean(activeSection.value.pages && activeSection.value.pages.length),
);

const hasToc = computed(() => Boolean(activeSection.value.toc));

const isEmbed = computed(() => route.query.embed === "1");

const layoutClass = computed(() => {
  if (hasSidebar.value && hasToc.value) return "with-sidebar with-toc";
  if (hasSidebar.value) return "with-sidebar";
  if (hasToc.value) return "with-toc";
  return "with-none";
});

// Strip emoji prefixes (flags etc.) and any inline HTML (e.g. SVG flag <img>
// tags used when no Unicode flag emoji exists) for the document title.
const stripEmoji = (s: string): string =>
  s
    .replace(/<[a-z][a-z0-9]*\b(?:[^>'"]|'[^']*'|"[^"]*")*>/gi, "")
    .replace(
      /[\p{Extended_Pictographic}\p{Regional_Indicator}‍️]/gu,
      "",
    )
    .trim();

useHead(() => {
  const section = activeSection.value;
  const page = activePage.value;
  if (page) {
    const pageTitle = stripEmoji(page.title);
    return {
      title: `${pageTitle} · ${section.title}`,
      description: isLeaderboard.value
        ? `${pageTitle} leaderboard: ranks language models across EuroEval tasks for this language.`
        : `${pageTitle} — ${section.title} documentation for the EuroEval benchmark.`,
      path: route.fullPath,
    };
  }
  if (section.id !== defaultSection.id) {
    return {
      title: section.title,
      description: `${section.title} section of the EuroEval benchmark documentation.`,
      path: route.fullPath,
    };
  }
  return { path: route.fullPath };
});
</script>

<template>
  <div class="app" :class="{ embed: isEmbed }">
    <TopHeader v-if="!isEmbed" />
    <SectionTabs v-if="!isEmbed" :active-id="activeSection.id" />
    <div class="page" :class="layoutClass">
      <SideNav
        v-if="hasSidebar && !isEmbed"
        :section="activeSection"
        :active-page-slug="pageSlug"
      />
      <main class="content-wrap">
        <LeaderboardView
          v-if="isLeaderboard && activePage"
          :stem="activePage.csv!"
          :title="activePage.title"
        />
        <EvaluationQueueView
          v-else-if="customView === 'EvaluationQueue'"
        />
        <MarkdownRenderer
          v-else-if="mdPath"
          :path="mdPath"
          :section="activeSection"
          :page-slug="pageSlug"
        />
        <div v-else class="error">Page not found.</div>
      </main>
      <TableOfContents
        v-if="hasToc && mdPath && !isLeaderboard && !isEmbed"
        :path="mdPath"
      />
    </div>
    <footer v-if="!isEmbed" class="site-footer">
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

/* Embed mode: stripped-down layout for iframing. */
.app.embed {
  min-height: auto;
}

.app.embed .page {
  padding: 0.75rem 1rem;
  max-width: none;
  grid-template-columns: 1fr !important;
}

.app.embed .content-wrap {
  padding: 0;
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
  min-height: calc(100vh - 120px);
}

.page.with-sidebar {
  grid-template-columns: 240px 1fr;
}

.page.with-toc {
  grid-template-columns: 1fr 220px;
}

.page.with-sidebar.with-toc {
  grid-template-columns: 240px 1fr 220px;
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
  .page.with-sidebar.with-toc {
    grid-template-columns: 240px 1fr;
  }
}

@media (max-width: 768px) {
  .page.with-sidebar,
  .page.with-sidebar.with-toc {
    grid-template-columns: 1fr;
  }
}
</style>
