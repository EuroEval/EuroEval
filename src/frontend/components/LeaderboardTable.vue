<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { Column, LeaderboardTable, Row } from "@/leaderboard";

const props = defineProps<{
  table: LeaderboardTable;
}>();

type FilterValue = string;
const colFilters = ref<Record<string, FilterValue>>({});
const globalSearch = ref("");
const sortBy = ref<{ index: number; dir: "asc" | "desc" } | null>(null);
const page = ref(1);
const pageSize = 10;

// Default sort: rank ascending if present.
watch(
  () => props.table,
  () => {
    colFilters.value = {};
    page.value = 1;
    const rankIdx = props.table.columns.findIndex(
      (c) => c.key.toLowerCase() === "rank",
    );
    sortBy.value = rankIdx >= 0 ? { index: rankIdx, dir: "asc" } : null;
  },
  { immediate: true },
);

const passesColumnFilter = (cellText: string, filter: string, col: Column) => {
  if (!filter) return true;
  if (col.kind === "icon") {
    // For icon columns, exact match against the filter value.
    return cellText === filter;
  }
  return cellText.toLowerCase().includes(filter.toLowerCase());
};

const filteredRows = computed<Row[]>(() => {
  const cols = props.table.columns;
  const q = globalSearch.value.trim().toLowerCase();
  return props.table.rows.filter((row) => {
    // Per-column filters.
    for (let i = 0; i < cols.length; i++) {
      const filter = colFilters.value[cols[i].key];
      if (filter && !passesColumnFilter(row.cells[i].text, filter, cols[i])) {
        return false;
      }
    }
    // Global search.
    if (q) {
      const hit = row.cells.some((c) => c.text.toLowerCase().includes(q));
      if (!hit) return false;
    }
    return true;
  });
});

const sortedRows = computed<Row[]>(() => {
  const rows = filteredRows.value;
  const s = sortBy.value;
  if (!s) return rows;
  const idx = s.index;
  const dir = s.dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const ak = a.cells[idx].sortKey;
    const bk = b.cells[idx].sortKey;
    if (typeof ak === "number" && typeof bk === "number") {
      return (ak - bk) * dir;
    }
    return String(ak).localeCompare(String(bk)) * dir;
  });
});

const pageCount = computed(() =>
  Math.max(1, Math.ceil(sortedRows.value.length / pageSize)),
);

watch(pageCount, (n) => {
  if (page.value > n) page.value = n;
});

const pagedRows = computed<Row[]>(() => {
  const start = (page.value - 1) * pageSize;
  return sortedRows.value.slice(start, start + pageSize);
});

// How many filler rows to render after the data so each page has a constant
// height — keeps the pager controls from jumping when the last page is short.
const fillerRowCount = computed(() => {
  if (sortedRows.value.length === 0) return 0;
  return Math.max(0, pageSize - pagedRows.value.length);
});

// --- Display helpers ------------------------------------------------------

const COMPACT_COLS = new Set(["parameters", "vocabulary", "context"]);

const PERCENT_COLS = new Set(["european values"]);

const TICK_CROSS_COLS = new Set([
  "commercial",
  "merge",
  "open",
  "trained from scratch",
]);

const formatCompact = (n: number): string => {
  const abs = Math.abs(n);
  if (abs >= 1e12) return Math.round(n / 1e12) + "T";
  if (abs >= 1e9) return Math.round(n / 1e9) + "B";
  if (abs >= 1e6) return Math.round(n / 1e6) + "M";
  if (abs >= 1e3) return Math.round(n / 1e3) + "K";
  return String(Math.round(n));
};

const isCompactCol = (col: Column) =>
  col.kind === "number" && COMPACT_COLS.has(col.key.toLowerCase());

const isPercentCol = (col: Column) =>
  col.kind === "number" && PERCENT_COLS.has(col.key.toLowerCase());

const isTickCrossCol = (col: Column) =>
  col.kind === "icon" && TICK_CROSS_COLS.has(col.key.toLowerCase());

const isRankCol = (col: Column) => col.key.toLowerCase() === "rank";

const cellDisplayHtml = (cell: { html: string; text: string; sortKey: number | string }, col: Column): string => {
  if (isCompactCol(col)) {
    if (typeof cell.sortKey === "number" && Number.isFinite(cell.sortKey)) {
      return formatCompact(cell.sortKey);
    }
    return cell.text || "?";
  }
  if (isPercentCol(col)) {
    if (typeof cell.sortKey === "number" && Number.isFinite(cell.sortKey)) {
      return `${Math.round(cell.sortKey)}%`;
    }
    return cell.text || "?";
  }
  return cell.html;
};

const tickCrossClass = (text: string): string => {
  switch (text) {
    case "✓":
    case "(✓)":
      return "tc-good";
    case "✗":
    case "(✗)":
      return "tc-bad";
    default:
      return "tc-unknown";
  }
};

// --- Rank heatmap ---------------------------------------------------------

// Fixed three-stop scale: 1.0 → green, 2.5 → yellow, 10.0 → red.
// Sentinel-ranked rows (text "-", "?", "") get no background.
const RANK_GREEN: [number, number, number] = [60, 175, 90];
const RANK_YELLOW: [number, number, number] = [230, 200, 60];
const RANK_RED: [number, number, number] = [220, 80, 80];

const blendRgb = (
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] => [
  Math.round(a[0] + (b[0] - a[0]) * t),
  Math.round(a[1] + (b[1] - a[1]) * t),
  Math.round(a[2] + (b[2] - a[2]) * t),
];

const rankHeatmapStyle = (
  cell: { text: string; sortKey: number | string },
): Record<string, string> => {
  if (cell.text === "-" || cell.text === "?" || cell.text === "") return {};
  const v = cell.sortKey;
  if (typeof v !== "number" || !Number.isFinite(v)) return {};

  let rgb: [number, number, number];
  if (v <= 1) {
    rgb = RANK_GREEN;
  } else if (v <= 2.5) {
    rgb = blendRgb(RANK_GREEN, RANK_YELLOW, (v - 1) / 1.5);
  } else if (v <= 10) {
    rgb = blendRgb(RANK_YELLOW, RANK_RED, (v - 2.5) / 7.5);
  } else {
    rgb = RANK_RED;
  }
  return {
    backgroundColor: `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.6)`,
  };
};

const toggleSort = (idx: number) => {
  if (sortBy.value?.index === idx) {
    sortBy.value =
      sortBy.value.dir === "asc"
        ? { index: idx, dir: "desc" }
        : null;
  } else {
    sortBy.value = { index: idx, dir: "asc" };
  }
};

const resetFilters = () => {
  colFilters.value = {};
  globalSearch.value = "";
  page.value = 1;
};
</script>

<template>
  <div class="leaderboard">
    <div class="lb-toolbar">
      <input
        v-model="globalSearch"
        class="lb-search"
        type="search"
        placeholder="Search in table"
        aria-label="Search in table"
        @input="page = 1"
      />
      <button class="lb-reset" type="button" @click="resetFilters">
        Reset filters
      </button>
      <div class="lb-pageinfo">
        Page
        <select v-model.number="page" aria-label="Page">
          <option v-for="n in pageCount" :key="n" :value="n">{{ n }}</option>
        </select>
        of {{ pageCount }} · {{ sortedRows.length }} rows
      </div>
    </div>

    <div class="lb-scroll">
      <table class="lb-table">
        <thead>
          <tr>
            <th
              v-for="(col, i) in table.columns"
              :key="col.key + i"
              :class="['col', `kind-${col.kind}`]"
              @click="toggleSort(i)"
            >
              <span class="th-label" v-html="col.titleHtml || col.title" />
              <span class="th-sort">
                <template v-if="sortBy?.index === i">
                  {{ sortBy.dir === "asc" ? "▲" : "▼" }}
                </template>
              </span>
            </th>
          </tr>
          <tr class="filter-row">
            <th
              v-for="(col, i) in table.columns"
              :key="`f-${col.key}-${i}`"
              @click.stop
            >
              <input
                v-if="i === 0"
                v-model="colFilters[col.key]"
                type="search"
                class="lb-filter"
                placeholder="filter"
                :aria-label="`Filter ${col.title}`"
                @input="page = 1"
              />
              <select
                v-else-if="col.kind === 'icon' && col.distinctValues"
                v-model="colFilters[col.key]"
                class="lb-filter"
                :aria-label="`Filter ${col.title}`"
                @change="page = 1"
              >
                <option value="">any</option>
                <option v-for="v in col.distinctValues" :key="v" :value="v">
                  {{ v }}
                </option>
              </select>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, ri) in pagedRows" :key="ri">
            <td
              v-for="(cell, ci) in row.cells"
              :key="ci"
              :class="[
                'cell',
                `kind-${table.columns[ci].kind}`,
                isRankCol(table.columns[ci]) ? 'cell-rank' : '',
                isTickCrossCol(table.columns[ci]) ? `cell-tc ${tickCrossClass(cell.text)}` : '',
              ]"
              :style="isRankCol(table.columns[ci]) ? rankHeatmapStyle(cell) : undefined"
              :title="ci === 0 ? cell.text : undefined"
            >
              <span v-html="cellDisplayHtml(cell, table.columns[ci])" />
            </td>
          </tr>
          <tr v-if="pagedRows.length === 0">
            <td :colspan="table.columns.length" class="empty">
              No rows match these filters.
            </td>
          </tr>
          <!-- Filler rows so every page has the same height, keeping the
               pager controls in a fixed position. -->
          <tr
            v-for="i in fillerRowCount"
            :key="`filler-${i}`"
            class="filler-row"
            aria-hidden="true"
          >
            <td :colspan="table.columns.length"></td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="lb-pager">
      <button
        type="button"
        :disabled="page <= 1"
        @click="page = Math.max(1, page - 1)"
      >
        ‹ Prev
      </button>
      <span>Page {{ page }} of {{ pageCount }}</span>
      <button
        type="button"
        :disabled="page >= pageCount"
        @click="page = Math.min(pageCount, page + 1)"
      >
        Next ›
      </button>
    </div>
  </div>
</template>

<style scoped>
.leaderboard {
  font-size: 0.85rem;
}

.lb-toolbar {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}

.lb-search {
  flex: 1;
  min-width: 200px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  color: var(--color-text);
  padding: 0.4rem 0.6rem;
  border-radius: 4px;
  font: inherit;
}

.lb-search:focus {
  outline: 2px solid var(--color-link);
  outline-offset: -1px;
}

.lb-reset {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 0.4rem 0.7rem;
  border-radius: 4px;
  cursor: pointer;
}

.lb-reset:hover {
  background: var(--color-border);
}

.lb-pageinfo {
  color: var(--color-muted);
  font-size: 0.8rem;
}

.lb-pageinfo select {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.15rem 0.3rem;
  margin: 0 0.25rem;
}

.lb-scroll {
  overflow-x: auto;
  border: 1px solid var(--color-border);
  border-radius: 6px;
}

.lb-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}

.lb-table th,
.lb-table td {
  border-bottom: 1px solid var(--color-border);
  padding: 0.45rem 0.6rem;
  text-align: center;
  vertical-align: middle;
  white-space: nowrap;
}

/* Force every data row to the same height so the table is a fixed size
   across pages. The Model column truncates with an ellipsis instead of
   wrapping; the full text is available via the cell title tooltip. */
.lb-table tbody tr {
  height: 44px;
}

/* First column (Model) stays left-aligned and ellipsis-truncates. */
.lb-table th:first-child,
.lb-table td:first-child {
  text-align: left;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.lb-table td:first-child > span {
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}

.lb-table thead th {
  background: var(--color-surface);
  cursor: pointer;
  user-select: none;
  font-weight: 600;
  position: sticky;
  top: 0;
}

.lb-table thead .filter-row th {
  position: sticky;
  top: 36px;
  background: var(--color-surface);
  cursor: default;
  font-weight: normal;
  padding: 0.3rem 0.4rem;
}

.th-label :deep(a) {
  color: inherit;
  text-decoration: none;
}

.th-label :deep(a:hover) {
  text-decoration: underline;
}

.th-sort {
  margin-left: 0.4rem;
  font-size: 0.7rem;
  color: var(--color-link);
}

.lb-filter {
  width: 100%;
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 3px;
  padding: 0.2rem 0.35rem;
  font: inherit;
  font-size: 0.78rem;
}

.lb-filter:focus {
  outline: 1px solid var(--color-link);
  outline-offset: -1px;
}

.cell.kind-model {
  min-width: 220px;
}

.cell.kind-number,
.cell.kind-score {
  font-variant-numeric: tabular-nums;
}

.cell-rank {
  font-weight: 500;
}

.filler-row td {
  /* Approximate average single-line data-row height so the table stays the
     same overall size regardless of how many real rows fill the last page. */
  height: 44px;
  border-bottom-color: transparent;
}

.cell-tc.tc-good {
  color: var(--color-icon-good);
  font-weight: 600;
}

.cell-tc.tc-bad {
  color: var(--color-icon-bad);
  font-weight: 600;
}

.cell-tc.tc-unknown {
  color: var(--color-icon-unknown);
}

.cell :deep(a) {
  color: var(--color-link);
  text-decoration: none;
}

.cell :deep(a:hover) {
  text-decoration: underline;
}

.empty {
  text-align: center;
  color: var(--color-muted);
  padding: 1.5rem;
}

.lb-pager {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.75rem;
  font-size: 0.85rem;
}

.lb-pager button {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  padding: 0.3rem 0.7rem;
  border-radius: 4px;
  cursor: pointer;
}

.lb-pager button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.lb-pager button:not(:disabled):hover {
  background: var(--color-border);
}
</style>
