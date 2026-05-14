<script setup lang="ts">
import { computed, ref } from "vue";
import type { LeaderboardTable } from "@/leaderboard";

const props = defineProps<{
  table: LeaderboardTable;
}>();

interface Point {
  x: number;
  y: number;
  label: string;
  icon: string;
}

const colIndex = (key: string) =>
  props.table.columns.findIndex((c) => c.key.toLowerCase() === key.toLowerCase());

const xAxisOptions = computed(() =>
  props.table.columns
    .filter((c) => c.kind === "number" && c.key.toLowerCase() !== "rank")
    .map((c) => c.key),
);

const xKey = ref<string>("");

// Default x-axis: Parameters when present, else first numeric column.
const ensureDefaultX = () => {
  if (xKey.value && xAxisOptions.value.includes(xKey.value)) return;
  const opts = xAxisOptions.value;
  if (opts.length === 0) {
    xKey.value = "";
    return;
  }
  xKey.value = opts.find((k) => k.toLowerCase() === "parameters") ?? opts[0];
};

ensureDefaultX();

const xIdx = computed(() => colIndex(xKey.value));
const yIdx = computed(() => colIndex("rank"));
const modelIdx = computed(() => colIndex("model"));
const typeIdx = computed(() => colIndex("type"));

const points = computed<Point[]>(() => {
  const xi = xIdx.value;
  const yi = yIdx.value;
  const mi = modelIdx.value;
  const ti = typeIdx.value;
  if (xi < 0 || yi < 0) return [];

  const out: Point[] = [];
  for (const row of props.table.rows) {
    const xk = row.cells[xi].sortKey;
    const yk = row.cells[yi].sortKey;
    if (typeof xk !== "number" || !Number.isFinite(xk)) continue;
    if (typeof yk !== "number" || !Number.isFinite(yk)) continue;
    out.push({
      x: xk,
      y: yk,
      label: mi >= 0 ? row.cells[mi].text : "",
      icon: ti >= 0 ? row.cells[ti].text : "",
    });
  }
  return out;
});

// SVG plot geometry.
const width = 800;
const height = 460;
const margin = { top: 20, right: 20, bottom: 48, left: 60 };

const xMinMax = computed(() => {
  const xs = points.value.map((p) => p.x);
  if (xs.length === 0) return { min: 0, max: 1 };
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  return min === max ? { min: min - 1, max: max + 1 } : { min, max };
});

const yMinMax = computed(() => {
  const ys = points.value.map((p) => p.y);
  if (ys.length === 0) return { min: 0, max: 1 };
  const min = Math.min(...ys);
  const max = Math.max(...ys);
  return min === max ? { min: min - 0.5, max: max + 0.5 } : { min, max };
});

const useLogX = computed(
  () => xKey.value.toLowerCase() === "parameters" && xMinMax.value.min > 0,
);

const xScale = (v: number) => {
  const { min, max } = xMinMax.value;
  const useLog = useLogX.value;
  const lo = useLog ? Math.log10(Math.max(min, 1)) : min;
  const hi = useLog ? Math.log10(Math.max(max, 1)) : max;
  const val = useLog ? Math.log10(Math.max(v, 1)) : v;
  const t = (val - lo) / (hi - lo || 1);
  return margin.left + t * (width - margin.left - margin.right);
};

const yScale = (v: number) => {
  // Lower rank = better, so invert.
  const { min, max } = yMinMax.value;
  const t = (v - min) / (max - min || 1);
  return margin.top + t * (height - margin.top - margin.bottom);
};

const formatX = (v: number): string => {
  if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(1) + "T";
  if (Math.abs(v) >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toFixed(1).replace(/\.0$/, "");
};

const xTicks = computed<number[]>(() => {
  const { min, max } = xMinMax.value;
  if (useLogX.value) {
    const ticks: number[] = [];
    const lo = Math.floor(Math.log10(Math.max(min, 1)));
    const hi = Math.ceil(Math.log10(Math.max(max, 1)));
    for (let i = lo; i <= hi; i++) ticks.push(Math.pow(10, i));
    return ticks.filter((t) => t >= min && t <= max);
  }
  // Linear: 5 evenly spaced.
  const ticks: number[] = [];
  for (let i = 0; i <= 5; i++) ticks.push(min + (max - min) * (i / 5));
  return ticks;
});

const yTicks = computed<number[]>(() => {
  const { min, max } = yMinMax.value;
  const ticks: number[] = [];
  for (let i = 0; i <= 5; i++) ticks.push(min + (max - min) * (i / 5));
  return ticks;
});

const hovered = ref<Point | null>(null);
const hoverPos = ref<{ x: number; y: number }>({ x: 0, y: 0 });

const onMove = (e: MouseEvent, p: Point) => {
  hovered.value = p;
  const target = e.currentTarget as SVGElement;
  const svg = target.ownerSVGElement as SVGSVGElement;
  const rect = svg.getBoundingClientRect();
  hoverPos.value = { x: e.clientX - rect.left, y: e.clientY - rect.top };
};

const onLeave = () => {
  hovered.value = null;
};
</script>

<template>
  <div class="scatter">
    <div class="scatter-toolbar">
      <label>
        X-axis:
        <select v-model="xKey">
          <option v-for="opt in xAxisOptions" :key="opt" :value="opt">
            {{ opt }}
          </option>
        </select>
      </label>
      <span class="scatter-help">
        Y-axis: Rank (lower is better). {{ points.length }} models plotted.
      </span>
    </div>

    <div class="scatter-wrap">
      <svg
        :viewBox="`0 0 ${width} ${height}`"
        preserveAspectRatio="xMidYMid meet"
        class="scatter-svg"
      >
        <!-- Axes -->
        <line
          :x1="margin.left"
          :x2="width - margin.right"
          :y1="height - margin.bottom"
          :y2="height - margin.bottom"
          class="axis"
        />
        <line
          :x1="margin.left"
          :x2="margin.left"
          :y1="margin.top"
          :y2="height - margin.bottom"
          class="axis"
        />

        <!-- Gridlines + Y ticks -->
        <g class="grid">
          <line
            v-for="(t, i) in yTicks"
            :key="`yg-${i}`"
            :x1="margin.left"
            :x2="width - margin.right"
            :y1="yScale(t)"
            :y2="yScale(t)"
          />
          <text
            v-for="(t, i) in yTicks"
            :key="`yt-${i}`"
            class="tick-label"
            :x="margin.left - 8"
            :y="yScale(t)"
            dominant-baseline="middle"
            text-anchor="end"
          >
            {{ t.toFixed(2) }}
          </text>
        </g>

        <!-- X ticks -->
        <g class="grid">
          <line
            v-for="(t, i) in xTicks"
            :key="`xg-${i}`"
            :x1="xScale(t)"
            :x2="xScale(t)"
            :y1="margin.top"
            :y2="height - margin.bottom"
          />
          <text
            v-for="(t, i) in xTicks"
            :key="`xt-${i}`"
            class="tick-label"
            :x="xScale(t)"
            :y="height - margin.bottom + 16"
            text-anchor="middle"
          >
            {{ formatX(t) }}
          </text>
        </g>

        <!-- Axis labels -->
        <text
          class="axis-label"
          :x="width / 2"
          :y="height - 6"
          text-anchor="middle"
        >
          {{ xKey }}{{ useLogX ? " (log)" : "" }}
        </text>
        <text
          class="axis-label"
          :transform="`translate(14, ${height / 2}) rotate(-90)`"
          text-anchor="middle"
        >
          Rank
        </text>

        <!-- Points -->
        <g class="points">
          <circle
            v-for="(p, i) in points"
            :key="`p-${i}`"
            :cx="xScale(p.x)"
            :cy="yScale(p.y)"
            r="5"
            class="point"
            @mousemove="onMove($event, p)"
            @mouseleave="onLeave"
          />
        </g>
      </svg>

      <div
        v-if="hovered"
        class="tooltip"
        :style="{ left: `${hoverPos.x + 12}px`, top: `${hoverPos.y + 12}px` }"
      >
        <div class="tt-model">{{ hovered.icon }} {{ hovered.label }}</div>
        <div class="tt-meta">
          {{ xKey }}: {{ formatX(hovered.x) }} · Rank: {{ hovered.y.toFixed(2) }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.scatter {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.scatter-toolbar {
  display: flex;
  align-items: center;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--color-muted);
  flex-wrap: wrap;
}

.scatter-toolbar select {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.2rem 0.4rem;
  margin-left: 0.4rem;
  font: inherit;
}

.scatter-help {
  font-size: 0.8rem;
}

.scatter-wrap {
  position: relative;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 0.5rem;
}

.scatter-svg {
  width: 100%;
  height: auto;
  display: block;
}

.axis {
  stroke: var(--color-border);
  stroke-width: 1;
}

.grid line {
  stroke: var(--color-border);
  stroke-dasharray: 2 4;
  opacity: 0.4;
}

.tick-label,
.axis-label {
  fill: var(--color-muted);
  font-size: 11px;
}

.point {
  fill: var(--color-link);
  fill-opacity: 0.75;
  stroke: var(--color-bg);
  stroke-width: 1.2;
  cursor: pointer;
  transition: r 0.1s ease;
}

.point:hover {
  fill-opacity: 1;
  r: 7;
}

.tooltip {
  position: absolute;
  pointer-events: none;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.4rem 0.6rem;
  font-size: 0.78rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  max-width: 320px;
  z-index: 5;
}

.tt-model {
  font-weight: 500;
  margin-bottom: 0.15rem;
}

.tt-meta {
  color: var(--color-muted);
}
</style>
