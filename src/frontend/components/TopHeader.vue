<script setup lang="ts">
import { navConfig } from "@/nav";
import { searchAll, type SearchResult } from "@/search";
import { toggleDrawer } from "@/drawer";
import { ref, onMounted, watch, computed, nextTick } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();

const theme = ref<"dark" | "light">(
  (localStorage.getItem("theme") as "dark" | "light") || "dark",
);

const applyTheme = (t: "dark" | "light") => {
  document.documentElement.setAttribute("data-theme", t);
};

onMounted(() => applyTheme(theme.value));
watch(theme, (t) => {
  applyTheme(t);
  localStorage.setItem("theme", t);
});

const toggleTheme = () => {
  theme.value = theme.value === "dark" ? "light" : "dark";
};

const ghUrl = `https://github.com/${navConfig.github.repo}`;

// GitHub stats
interface GhStats {
  stars: number;
  forks: number;
}
const ghStats = ref<GhStats | null>(null);

const formatCount = (n: number): string => {
  if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
  return String(n);
};

onMounted(async () => {
  const cacheKey = `gh-stats:${navConfig.github.repo}`;
  const cached = sessionStorage.getItem(cacheKey);
  if (cached) {
    try {
      ghStats.value = JSON.parse(cached);
      return;
    } catch {
      /* fall through */
    }
  }
  try {
    const res = await fetch(
      `https://api.github.com/repos/${navConfig.github.repo}`,
    );
    if (!res.ok) return;
    const data = await res.json();
    const stats = {
      stars: data.stargazers_count ?? 0,
      forks: data.forks_count ?? 0,
    };
    ghStats.value = stats;
    sessionStorage.setItem(cacheKey, JSON.stringify(stats));
  } catch {
    /* ignore network errors */
  }
});

// Search
const searchQuery = ref("");
const searchOpen = ref(false);
const searchInput = ref<HTMLInputElement | null>(null);
const activeIdx = ref(-1);

const results = computed<SearchResult[]>(() => searchAll(searchQuery.value));

watch(results, () => {
  activeIdx.value = -1;
});

const onFocus = () => {
  searchOpen.value = true;
};

const onBlur = () => {
  // Delay so click on result fires before closing.
  setTimeout(() => {
    searchOpen.value = false;
  }, 150);
};

const goTo = (url: string) => {
  router.push(url);
  searchQuery.value = "";
  searchOpen.value = false;
  searchInput.value?.blur();
};

const onKeydown = (e: KeyboardEvent) => {
  if (!searchOpen.value || results.value.length === 0) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeIdx.value = (activeIdx.value + 1) % results.value.length;
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeIdx.value =
      (activeIdx.value - 1 + results.value.length) % results.value.length;
  } else if (e.key === "Enter") {
    const pick =
      activeIdx.value >= 0 ? results.value[activeIdx.value] : results.value[0];
    if (pick) {
      e.preventDefault();
      goTo(pick.entry.url);
    }
  } else if (e.key === "Escape") {
    searchOpen.value = false;
    searchInput.value?.blur();
  }
};

// Cmd/Ctrl+K focuses search.
const onGlobalKey = (e: KeyboardEvent) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    nextTick(() => searchInput.value?.focus());
  }
};

onMounted(() => {
  window.addEventListener("keydown", onGlobalKey);
});
</script>

<template>
  <header class="top-header">
    <div class="header-inner">
      <button
        class="hamburger"
        aria-label="Open navigation"
        @click="toggleDrawer"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 6h18v2H3zm0 5h18v2H3zm0 5h18v2H3z" fill="currentColor" />
        </svg>
      </button>

      <router-link class="brand" to="/">
        <svg class="brand-logo" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M22 21H2V3h2v16h2v-9h4v9h2V6h4v13h2v-5h4z" />
        </svg>
        <span class="brand-name">EuroEval</span>
      </router-link>

      <div class="header-actions">
      <button
        class="theme-toggle"
        :aria-label="theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'"
        @click="toggleTheme"
      >
        <svg v-if="theme === 'dark'" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zm0-5v2m0 16v2m10-10h-2M4 12H2m15.07-7.07-1.41 1.41M6.34 17.66l-1.41 1.41m12.73 0-1.41-1.41M6.34 6.34 4.93 4.93" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/>
        </svg>
        <svg v-else viewBox="0 0 24 24" aria-hidden="true">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" fill="currentColor"/>
        </svg>
      </button>

      <div class="search" @keydown="onKeydown">
        <svg class="search-icon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M15.5 14h-.79l-.28-.27A6.5 6.5 0 1 0 13 15.5l.27.28v.79l5 4.99L19.49 20l-4.99-5zM10 14a4 4 0 1 1 0-8 4 4 0 0 1 0 8z" fill="currentColor"/>
        </svg>
        <input
          ref="searchInput"
          v-model="searchQuery"
          type="search"
          placeholder="Search"
          aria-label="Search"
          @focus="onFocus"
          @blur="onBlur"
        />
        <div
          v-if="searchOpen && searchQuery.length >= 2"
          class="search-results"
        >
          <div v-if="results.length === 0" class="search-empty">
            No results.
          </div>
          <a
            v-for="(r, i) in results"
            :key="r.entry.url + i"
            class="search-result"
            :class="{ active: i === activeIdx }"
            href="#"
            @mousedown.prevent="goTo(r.entry.url)"
          >
            <div class="sr-title">
              <span class="sr-section">{{ r.entry.sectionTitle }}</span>
              <span class="sr-sep">›</span>
              <span v-html="r.entry.pageTitle" />
            </div>
            <div class="sr-snippet">{{ r.snippet }}</div>
          </a>
        </div>
      </div>

      <a class="gh-card" :href="ghUrl" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" class="gh-icon" aria-hidden="true">
          <path fill="currentColor" d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.55v-2.07c-3.2.7-3.87-1.36-3.87-1.36-.52-1.34-1.28-1.7-1.28-1.7-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.07 0 0 .97-.31 3.18 1.18a11.05 11.05 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.6.23 2.78.11 3.07.74.81 1.19 1.84 1.19 3.1 0 4.43-2.69 5.41-5.25 5.69.41.36.78 1.07.78 2.16v3.2c0 .31.21.66.8.55C20.21 21.39 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z"/>
        </svg>
        <span class="gh-meta">
          <span class="gh-repo">EuroEval</span>
          <span class="gh-row">
            <span class="gh-version">{{ navConfig.github.version }}</span>
            <span v-if="ghStats" class="gh-stats">
            <span class="gh-stat" aria-label="Stars">
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path fill="currentColor" d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.61L7.327.667A.75.75 0 0 1 8 .25z"/>
              </svg>
              {{ formatCount(ghStats.stars) }}
            </span>
            <span class="gh-stat" aria-label="Forks">
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path fill="currentColor" d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0zM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0z"/>
              </svg>
              {{ formatCount(ghStats.forks) }}
            </span>
          </span>
          </span>
        </span>
      </a>
      </div>
    </div>
  </header>
</template>

<style scoped>
.top-header {
  background: var(--color-header-bg);
  color: var(--color-header-text);
  position: sticky;
  top: 0;
  z-index: 10;
}

.header-inner {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0.75rem 2rem;
  display: flex;
  align-items: center;
  gap: 1.25rem;
  min-height: 64px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  margin-left: auto;
}

.hamburger {
  background: transparent;
  border: 0;
  color: inherit;
  cursor: pointer;
  padding: 0.25rem;
  display: none;
  align-items: center;
}

.hamburger svg {
  width: 22px;
  height: 22px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: inherit;
  text-decoration: none;
}

.brand-logo {
  width: 22px;
  height: 22px;
  fill: currentColor;
}

.brand-name {
  font-size: 1.1rem;
  font-weight: 500;
}

.theme-toggle {
  background: transparent;
  border: 0;
  color: inherit;
  cursor: pointer;
  padding: 0.35rem;
  display: flex;
  align-items: center;
  border-radius: 50%;
  flex-shrink: 0;
}

.theme-toggle:hover {
  background: rgba(255, 255, 255, 0.12);
}

.theme-toggle svg {
  width: 20px;
  height: 20px;
}

.search {
  position: relative;
  width: 260px;
}

.search-icon {
  position: absolute;
  left: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  width: 18px;
  height: 18px;
  color: rgba(255, 255, 255, 0.7);
  pointer-events: none;
}

.search input {
  width: 100%;
  border: 0;
  background: rgba(255, 255, 255, 0.18);
  color: inherit;
  font: inherit;
  padding: 0.4rem 0.6rem 0.4rem 2rem;
  border-radius: 4px;
  outline: none;
}

.search input::placeholder {
  color: rgba(255, 255, 255, 0.7);
}

.search input:focus {
  background: rgba(255, 255, 255, 0.28);
}

.search-results {
  position: absolute;
  left: 0;
  right: 0;
  top: calc(100% + 4px);
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
  max-height: 70vh;
  overflow-y: auto;
  z-index: 20;
}

.search-empty {
  padding: 0.75rem 1rem;
  color: var(--color-muted);
  font-size: 0.85rem;
}

.search-result {
  display: block;
  padding: 0.6rem 0.9rem;
  text-decoration: none;
  color: inherit;
  border-bottom: 1px solid var(--color-border);
}

.search-result:last-child {
  border-bottom: 0;
}

.search-result:hover,
.search-result.active {
  background: var(--color-surface);
  text-decoration: none;
}

.sr-title {
  font-size: 0.9rem;
  font-weight: 500;
  margin-bottom: 0.2rem;
}

.sr-section {
  color: var(--color-link);
}

.sr-sep {
  margin: 0 0.35rem;
  color: var(--color-muted);
}

.sr-snippet {
  font-size: 0.8rem;
  color: var(--color-muted);
  line-height: 1.4;
}

.gh-card {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: inherit;
  text-decoration: none;
  font-size: 0.8rem;
  line-height: 1.2;
}

.gh-icon {
  width: 26px;
  height: 26px;
}

.gh-meta {
  display: flex;
  flex-direction: column;
}

.gh-repo {
  font-weight: 500;
}

.gh-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-top: 0.1rem;
}

.gh-version {
  opacity: 0.85;
  font-size: 0.72rem;
}

.gh-stats {
  display: flex;
  gap: 0.5rem;
  font-size: 0.72rem;
}

.gh-stat {
  display: inline-flex;
  align-items: center;
  gap: 0.2rem;
}

.gh-stat svg {
  width: 12px;
  height: 12px;
}

@media (max-width: 900px) {
  .gh-card .gh-meta {
    display: none;
  }
  .search {
    width: 200px;
  }
}

@media (max-width: 768px) {
  .hamburger {
    display: flex;
  }
  .header-inner {
    padding: 0.5rem 1rem;
    gap: 0.75rem;
    flex-wrap: wrap;
  }
  .search {
    width: 100%;
    order: 5;
    flex-basis: 100%;
    padding-bottom: 0.4rem;
  }
}
</style>
