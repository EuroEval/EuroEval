<script setup lang="ts">
import {
  computed,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import type { LeaderboardTable } from "@/leaderboard";
import { matchesQuery } from "@/filter";

type XAxis = "parameters" | "releaseDate";
type OptimizationDirection = "minimize" | "maximize";

const props = withDefaults(
  defineProps<{
    table: LeaderboardTable;
    xAxis?: XAxis;
  }>(),
  { xAxis: "parameters" },
);

type ModelKind = "instruct" | "reasoning" | "base" | "encoder" | "other";

interface MetaItem {
  label: string;
  value: string;
}

interface Point {
  x: number;
  xLabel: string;
  y: number;
  label: string;
  icon: string;
  kind: ModelKind;
  commercial: boolean;
  meta: MetaItem[];
}

const isNoWorse = (
  candidate: number,
  point: number,
  direction: OptimizationDirection,
) =>
  direction === "minimize" ? candidate <= point : candidate >= point;

const isStrictlyBetter = (
  candidate: number,
  point: number,
  direction: OptimizationDirection,
) =>
  direction === "minimize" ? candidate < point : candidate > point;

const dominates = (
  candidate: Point,
  point: Point,
  xDirection: OptimizationDirection,
) =>
  isNoWorse(candidate.x, point.x, xDirection) &&
  candidate.y <= point.y &&
  (isStrictlyBetter(candidate.x, point.x, xDirection) ||
    candidate.y < point.y);

// Columns to surface in the hover tooltip, in display order. We skip
// Model/Type/Rank (already shown elsewhere in the tooltip) and any
// per-dataset score columns (too noisy).
const META_KEYS: readonly string[] = [
  "parameters",
  "vocabulary",
  "context",
  "european values",
  "commercial",
  "merge",
  "open-weight",
  "trained from scratch",
];

const formatCompact = (n: number): string => {
  const abs = Math.abs(n);
  if (abs >= 1e12) return Math.round(n / 1e12) + "T";
  if (abs >= 1e9) return Math.round(n / 1e9) + "B";
  if (abs >= 1e6) return Math.round(n / 1e6) + "M";
  if (abs >= 1e3) return Math.round(n / 1e3) + "K";
  return String(Math.round(n));
};

const COMPACT_KEYS = new Set(["parameters", "vocabulary", "context"]);
const PERCENT_KEYS = new Set(["european values"]);

const formatMetaValue = (key: string, cell: { text: string; sortKey: number | string }): string => {
  if (cell.text === "-" || cell.text === "?" || cell.text === "") return "—";
  if (COMPACT_KEYS.has(key) && typeof cell.sortKey === "number" && Number.isFinite(cell.sortKey)) {
    return formatCompact(cell.sortKey);
  }
  if (PERCENT_KEYS.has(key) && typeof cell.sortKey === "number" && Number.isFinite(cell.sortKey)) {
    return `${Math.round(cell.sortKey)}%`;
  }
  return cell.text;
};

const KIND_FROM_ICON: Record<string, ModelKind> = {
  "📝": "instruct",
  "🤔": "reasoning",
  "🧠": "base",
  "🔍": "encoder",
};

const KIND_LABEL: Record<ModelKind, string> = {
  instruct: "Instruction-tuned",
  reasoning: "Reasoning",
  base: "Base generative",
  encoder: "Encoder",
  other: "Other",
};

const KIND_COLOR: Record<ModelKind, string> = {
  instruct: "#5b9dd9", // blue
  reasoning: "#f0a040", // amber
  base: "#5cb874", // green
  encoder: "#b07ad8", // purple
  other: "#9aa0a6", // muted
};

const colIndex = (key: string) =>
  props.table.columns.findIndex((c) => c.key.toLowerCase() === key.toLowerCase());

const xIdx = computed(() =>
  props.xAxis === "parameters" ? colIndex("parameters") : -1,
);
const yIdx = computed(() => colIndex("rank score"));
const modelIdx = computed(() => colIndex("model"));
const typeIdx = computed(() => colIndex("type"));
const commercialIdx = computed(() => colIndex("commercial"));

// For each metadata key we want to show, cache its column index (if present).
const metaIndices = computed<{ key: string; title: string; idx: number }[]>(() =>
  META_KEYS.map((key) => {
    const idx = props.table.columns.findIndex(
      (c) => c.key.toLowerCase() === key,
    );
    const title = idx >= 0 ? props.table.columns[idx].title : key;
    return { key, title, idx };
  }).filter((e) => e.idx >= 0),
);

const allPoints = computed<Point[]>(() => {
  const xi = xIdx.value;
  const yi = yIdx.value;
  const mi = modelIdx.value;
  const ti = typeIdx.value;
  const ci = commercialIdx.value;
  if ((props.xAxis === "parameters" && xi < 0) || yi < 0) return [];

  const out: Point[] = [];
  // Drop rows whose displayed value is a sentinel (e.g. "-", "?", "") —
  // these have a sort key like `-@@100` purely to push them to the bottom
  // of the sorted table, and shouldn't appear on the plot.
  const isRealValue = (text: string) =>
    text !== "" && text !== "-" && text !== "?" && text !== "??";

  for (const row of props.table.rows) {
    const yc = row.cells[yi];
    if (!isRealValue(yc.text)) continue;
    let xk: number;
    let xLabel: string;
    if (props.xAxis === "releaseDate") {
      if (!row.releaseDate) continue;
      xk = Date.parse(`${row.releaseDate}T00:00:00Z`);
      xLabel = row.releaseDate;
    } else {
      const xc = row.cells[xi];
      if (!isRealValue(xc.text) || typeof xc.sortKey !== "number") continue;
      xk = xc.sortKey;
      xLabel = formatCompact(xk);
    }
    const yk = yc.sortKey;
    if (!Number.isFinite(xk)) continue;
    if (typeof yk !== "number" || !Number.isFinite(yk)) continue;
    const icon = ti >= 0 ? row.cells[ti].text : "";
    const commercialText = ci >= 0 ? row.cells[ci].text : "";
    const meta: MetaItem[] = metaIndices.value.map(({ key, title, idx }) => ({
      label: title,
      value: formatMetaValue(key, row.cells[idx]),
    }));
    out.push({
      x: xk,
      xLabel,
      y: yk,
      label: mi >= 0 ? row.cells[mi].text : "",
      icon,
      kind: KIND_FROM_ICON[icon] ?? "other",
      commercial: commercialText === "✓",
      meta,
    });
  }
  return out;
});

// Legend uses only the kinds that actually appear in the data.
const presentKinds = computed<ModelKind[]>(() => {
  const seen = new Set<ModelKind>();
  for (const p of allPoints.value) seen.add(p.kind);
  return (["instruct", "reasoning", "base", "encoder", "other"] as const).filter(
    (k) => seen.has(k),
  );
});

const hasCommercial = computed(() =>
  allPoints.value.some((p) => p.commercial),
);
const hasNonCommercial = computed(() =>
  allPoints.value.some((p) => !p.commercial),
);

// Visibility toggles. All groups visible by default; clicking a legend
// item toggles its group off/on.
const hiddenKinds = ref<Set<ModelKind>>(new Set());
const hideCommercial = ref(false);
const hideNonCommercial = ref(false);
const searchQuery = ref("");
const paretoVisibility = ref<Record<XAxis, boolean>>({
  parameters: true,
  releaseDate: true,
});

const showPareto = computed(() => paretoVisibility.value[props.xAxis]);
const xOptimization: OptimizationDirection = "minimize";
const paretoDescription = computed(() =>
  props.xAxis === "parameters"
    ? "Lower parameter count and lower rank score are preferred."
    : "Earlier release date and lower rank score are preferred.",
);
const paretoToggleLabel = computed(
  () =>
    `${showPareto.value ? "Hide" : "Show"} Pareto curve. ` +
    paretoDescription.value,
);

const togglePareto = () => {
  paretoVisibility.value = {
    ...paretoVisibility.value,
    [props.xAxis]: !showPareto.value,
  };
};

const toggleKind = (k: ModelKind) => {
  if (hiddenKinds.value.has(k)) {
    hiddenKinds.value.delete(k);
  } else {
    hiddenKinds.value.add(k);
  }
  // Force reactivity (Set mutation isn't tracked).
  hiddenKinds.value = new Set(hiddenKinds.value);
};

const isKindHidden = (k: ModelKind) => hiddenKinds.value.has(k);

const visiblePoints = computed<Point[]>(() => {
  const query = searchQuery.value.trim();
  return allPoints.value.filter((p) => {
    if (hiddenKinds.value.has(p.kind)) return false;
    if (p.commercial && hideCommercial.value) return false;
    if (!p.commercial && hideNonCommercial.value) return false;
    if (query && !matchesQuery(p.label, query)) return false;
    return true;
  });
});

const paretoPoints = computed<Point[]>(() => {
  const points = visiblePoints.value.filter(
    (point) => Number.isFinite(point.x) && Number.isFinite(point.y),
  );
  return points.filter(
    (point) =>
      !points.some(
        (candidate) =>
          candidate !== point &&
          dominates(candidate, point, xOptimization),
      ),
  );
});

const paretoPointSet = computed(() => new Set(paretoPoints.value));
const chartAriaLabel = computed(() => {
  const xLabel =
    props.xAxis === "parameters" ? "Parameter count" : "Release date";
  const frontierLabel = showPareto.value
    ? `, ${paretoPoints.value.length} on the Pareto frontier`
    : "";
  return (
    `${xLabel} versus rank score for ${visiblePoints.value.length} models` +
    frontierLabel
  );
});

// SVG plot geometry.
const width = 800;
const height = 460;
const margin = { top: 20, right: 20, bottom: 48, left: 60 };

// Axis domains are computed from *all* points (not just the visible ones)
// so toggling groups on/off doesn't rescale the plot.
const rawXMinMax = computed(() => {
  const xs = allPoints.value.map((p) => p.x);
  if (xs.length === 0) return { min: 0, max: 1 };
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  if (min !== max) return { min, max };
  const singleValuePad =
    props.xAxis === "releaseDate" ? 180 * 24 * 60 * 60 * 1000 : 1;
  return { min: min - singleValuePad, max: max + singleValuePad };
});

const useLogX = computed(
  () => props.xAxis === "parameters" && rawXMinMax.value.min > 0,
);

// The x-domain is the current horizontal range, padded on both ends so
// the smallest and largest models sit inside the plot with breathing room
// rather than clipped against the axis edges. Padding is applied in the same
// space the axis uses (log10 for model sizes), so it reads as a constant visual
// margin on each side regardless of how wide the range is.
const X_PAD = 0.06; // fraction of the (transformed) range, per side
const baseXMinMax = computed(() => {
  const { min, max } = rawXMinMax.value;
  if (useLogX.value) {
    const lo = Math.log10(Math.max(min, 1));
    const hi = Math.log10(Math.max(max, 1));
    const pad = (hi - lo || 1) * X_PAD;
    return { min: Math.pow(10, lo - pad), max: Math.pow(10, hi + pad) };
  }
  const pad = (max - min || 1) * X_PAD;
  return { min: min - pad, max: max + pad };
});

// Fixed mean-rank-score domain so plots are visually comparable across
// languages and pages. Lower is better, so 1.0 is at the top, 5.5 at the
// bottom.
const baseYMinMax = computed(() => ({ min: 1.0, max: 5.5 }));

interface Viewport {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

const toXSpace = (value: number) =>
  useLogX.value ? Math.log10(Math.max(value, 1)) : value;
const fromXSpace = (value: number) =>
  useLogX.value ? Math.pow(10, value) : value;

const baseViewport = computed<Viewport>(() => ({
  xMin: toXSpace(baseXMinMax.value.min),
  xMax: toXSpace(baseXMinMax.value.max),
  yMin: baseYMinMax.value.min,
  yMax: baseYMinMax.value.max,
}));

const viewport = ref<Viewport>({ ...baseViewport.value });
const MAX_ZOOM = 24;
const clipId = `scatter-plot-clip-${Math.random().toString(36).slice(2)}`;

const resetZoom = () => {
  viewport.value = { ...baseViewport.value };
};

// A changed dataset or x-axis starts a new viewport rather than carrying a
// range that may no longer contain the newly selected data.
watch([baseViewport, () => props.xAxis], resetZoom);

const xMinMax = computed(() => ({
  min: fromXSpace(viewport.value.xMin),
  max: fromXSpace(viewport.value.xMax),
}));
const yMinMax = computed(() => ({
  min: viewport.value.yMin,
  max: viewport.value.yMax,
}));
const isZoomed = computed(() => {
  const base = baseViewport.value;
  const current = viewport.value;
  return (
    current.xMin !== base.xMin ||
    current.xMax !== base.xMax ||
    current.yMin !== base.yMin ||
    current.yMax !== base.yMax
  );
});

const plotWidth = width - margin.left - margin.right;
const plotHeight = height - margin.top - margin.bottom;

const xScale = (v: number) => {
  const { min, max } = xMinMax.value;
  const useLog = useLogX.value;
  const lo = useLog ? Math.log10(Math.max(min, 1)) : min;
  const hi = useLog ? Math.log10(Math.max(max, 1)) : max;
  const val = useLog ? Math.log10(Math.max(v, 1)) : v;
  const t = (val - lo) / (hi - lo || 1);
  return margin.left + t * plotWidth;
};

const yScale = (v: number) => {
  // Lower rank = better, so invert.
  const { min, max } = yMinMax.value;
  const t = (v - min) / (max - min || 1);
  return margin.top + t * plotHeight;
};

const paretoPolyline = computed(() => {
  const uniquePoints = new Map<string, Point>();
  for (const point of paretoPoints.value) {
    uniquePoints.set(`${point.x}:${point.y}`, point);
  }
  return [...uniquePoints.values()]
    .sort((a, b) => a.x - b.x || a.y - b.y)
    .map((point) => `${xScale(point.x)},${yScale(point.y)}`)
    .join(" ");
});

const formatX = (v: number): string => {
  if (props.xAxis === "releaseDate") {
    const date = new Date(v);
    const rangeYears =
      (xMinMax.value.max - xMinMax.value.min) /
      (365.25 * 24 * 60 * 60 * 1000);
    return rangeYears > 2
      ? String(date.getUTCFullYear())
      : date.toLocaleDateString("en", {
          month: "short",
          year: "numeric",
          timeZone: "UTC",
        });
  }
  if (Math.abs(v) >= 1e12) return (v / 1e12).toFixed(1) + "T";
  if (Math.abs(v) >= 1e9) return (v / 1e9).toFixed(1) + "B";
  if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toFixed(1).replace(/\.0$/, "");
};

interface Tick {
  value: number;
  major: boolean;
}

const xTicks = computed<Tick[]>(() => {
  const { min, max } = xMinMax.value;
  if (props.xAxis === "releaseDate") {
    const tickMin = min;
    const tickMax = max;
    const rangeYears =
      (tickMax - tickMin) / (365.25 * 24 * 60 * 60 * 1000);
    const ticks: Tick[] = [];
    if (rangeYears > 2) {
      const firstYear = new Date(tickMin).getUTCFullYear();
      const lastYear = new Date(tickMax).getUTCFullYear();
      const step = Math.max(1, Math.ceil((lastYear - firstYear) / 6));
      for (let year = firstYear; year <= lastYear; year += step) {
        const value = Date.UTC(year, 0, 1);
        if (value >= min && value <= max) ticks.push({ value, major: true });
      }
    } else {
      const first = new Date(tickMin);
      const last = new Date(tickMax);
      const firstMonth = first.getUTCFullYear() * 12 + first.getUTCMonth();
      const lastMonth = last.getUTCFullYear() * 12 + last.getUTCMonth();
      const step = Math.max(1, Math.ceil((lastMonth - firstMonth) / 6));
      for (let month = firstMonth; month <= lastMonth; month += step) {
        const value = Date.UTC(Math.floor(month / 12), month % 12, 1);
        if (value >= min && value <= max) ticks.push({ value, major: true });
      }
    }
    return ticks;
  }
  if (useLogX.value) {
    const ticks: Tick[] = [];
    const lo = Math.floor(Math.log10(Math.max(min, 1)));
    const hi = Math.ceil(Math.log10(Math.max(max, 1)));
    for (let i = lo; i <= hi; i++) {
      const base = Math.pow(10, i);
      ticks.push({ value: base, major: true });
      // Intra-decade minor labels at 2× and 5×.
      ticks.push({ value: 2 * base, major: false });
      ticks.push({ value: 5 * base, major: false });
    }
    return ticks.filter((t) => t.value >= min && t.value <= max);
  }
  // Linear: 10 evenly spaced.
  const ticks: Tick[] = [];
  for (let i = 0; i <= 10; i++) {
    ticks.push({
      value: min + (max - min) * (i / 10),
      major: i % 2 === 0,
    });
  }
  return ticks;
});

const niceTickStep = (range: number, targetCount: number) => {
  const roughStep = range / targetCount;
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const normalized = roughStep / magnitude;
  const multiplier =
    normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
};

const yTicks = computed<Tick[]>(() => {
  const { min, max } = yMinMax.value;
  const step = niceTickStep(max - min, 8);
  const first = Math.ceil(min / step) * step;
  const ticks: Tick[] = [];
  for (let value = first; value <= max + step * 1e-9; value += step) {
    ticks.push({ value, major: true });
  }
  return ticks;
});

const formatY = (value: number) => {
  const step = niceTickStep(yMinMax.value.max - yMinMax.value.min, 8);
  const decimals = Math.min(4, Math.max(1, Math.ceil(-Math.log10(step))));
  return value.toFixed(decimals);
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const clampRange = (
  min: number,
  max: number,
  baseMin: number,
  baseMax: number,
) => {
  const baseSpan = baseMax - baseMin;
  const span = clamp(max - min, baseSpan / MAX_ZOOM, baseSpan);
  let nextMin = min;
  let nextMax = min + span;
  if (nextMin < baseMin) {
    nextMin = baseMin;
    nextMax = baseMin + span;
  }
  if (nextMax > baseMax) {
    nextMax = baseMax;
    nextMin = baseMax - span;
  }
  return { min: nextMin, max: nextMax };
};

const clampViewport = (next: Viewport): Viewport => {
  const base = baseViewport.value;
  const x = clampRange(next.xMin, next.xMax, base.xMin, base.xMax);
  const y = clampRange(next.yMin, next.yMax, base.yMin, base.yMax);
  return { xMin: x.min, xMax: x.max, yMin: y.min, yMax: y.max };
};

const zoomRangeAt = (
  min: number,
  max: number,
  baseMin: number,
  baseMax: number,
  factor: number,
  ratio: number,
) => {
  const span = max - min;
  const nextSpan = clamp(
    span / factor,
    (baseMax - baseMin) / MAX_ZOOM,
    baseMax - baseMin,
  );
  const focal = min + ratio * span;
  return clampRange(
    focal - ratio * nextSpan,
    focal + (1 - ratio) * nextSpan,
    baseMin,
    baseMax,
  );
};

const zoomViewportAt = (
  start: Viewport,
  factor: number,
  xRatio: number,
  yRatio: number,
): Viewport => {
  const x = zoomRangeAt(
    start.xMin,
    start.xMax,
    baseViewport.value.xMin,
    baseViewport.value.xMax,
    factor,
    xRatio,
  );
  const y = zoomRangeAt(
    start.yMin,
    start.yMax,
    baseViewport.value.yMin,
    baseViewport.value.yMax,
    factor,
    yRatio,
  );
  return { xMin: x.min, xMax: x.max, yMin: y.min, yMax: y.max };
};

const chartPointFromClient = (
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
) => {
  const rect = svg.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return null;
  return {
    x: ((clientX - rect.left) / rect.width) * width,
    y: ((clientY - rect.top) / rect.height) * height,
  };
};

const chartPointFromEvent = (e: {
  clientX: number;
  clientY: number;
  currentTarget: EventTarget | null;
}) => {
  const svg = e.currentTarget as SVGSVGElement | null;
  return svg ? chartPointFromClient(svg, e.clientX, e.clientY) : null;
};

const plotRatios = (point: { x: number; y: number }) => ({
  x: clamp((point.x - margin.left) / plotWidth, 0, 1),
  y: clamp((point.y - margin.top) / plotHeight, 0, 1),
});

const onWheel = (e: WheelEvent) => {
  const point = chartPointFromEvent(e);
  if (!point) return;
  const deltaMultiplier =
    e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1;
  const delta = e.deltaY * deltaMultiplier;
  const factor = clamp(Math.exp(-delta * 0.002), 0.5, 2);
  const ratios = plotRatios(point);
  viewport.value = zoomViewportAt(viewport.value, factor, ratios.x, ratios.y);
};

interface ClientPoint {
  x: number;
  y: number;
}

const activePointers = new Map<number, ClientPoint>();
let panStart: {
  point: { x: number; y: number };
  viewport: Viewport;
} | null = null;
let pinchStart: {
  distance: number;
  midpoint: { x: number; y: number };
  focal: { x: number; y: number };
  viewport: Viewport;
} | null = null;
let suppressNextClick = false;

const pointerDistance = (a: ClientPoint, b: ClientPoint) =>
  Math.hypot(a.x - b.x, a.y - b.y);

const pointerMidpoint = (a: ClientPoint, b: ClientPoint): ClientPoint => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});

const onPointerDown = (e: PointerEvent) => {
  if (e.pointerType === "mouse" && e.button !== 0) return;
  if (activePointers.size === 0) suppressNextClick = false;
  const svg = e.currentTarget as SVGSVGElement;
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  svg.setPointerCapture?.(e.pointerId);

  if (activePointers.size === 1) {
    const target = e.target as Element | null;
    if (!target?.closest(".point")) {
      const point = chartPointFromEvent(e);
      if (point) panStart = { point, viewport: { ...viewport.value } };
    }
    return;
  }

  if (activePointers.size === 2) {
    panStart = null;
    const [first, second] = [...activePointers.values()];
    const midpoint = pointerMidpoint(first, second);
    const focal = chartPointFromClient(svg, midpoint.x, midpoint.y);
    if (focal) {
      pinchStart = {
        distance: Math.max(pointerDistance(first, second), 1),
        midpoint: focal,
        focal,
        viewport: { ...viewport.value },
      };
    }
  }
};

const onPointerMove = (e: PointerEvent) => {
  const previous = activePointers.get(e.pointerId);
  if (!previous) return;
  activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  const svg = e.currentTarget as SVGSVGElement;

  if (activePointers.size >= 2 && pinchStart) {
    const [first, second] = [...activePointers.values()];
    const midpoint = pointerMidpoint(first, second);
    const currentMidpoint = chartPointFromClient(svg, midpoint.x, midpoint.y);
    if (!currentMidpoint) return;
    const factor = pointerDistance(first, second) / pinchStart.distance;
    const ratios = plotRatios(pinchStart.focal);
    const zoomed = zoomViewportAt(
      pinchStart.viewport,
      factor,
      ratios.x,
      ratios.y,
    );
    const dx = currentMidpoint.x - pinchStart.midpoint.x;
    const dy = currentMidpoint.y - pinchStart.midpoint.y;
    viewport.value = clampViewport({
      xMin: zoomed.xMin - (dx / plotWidth) * (zoomed.xMax - zoomed.xMin),
      xMax: zoomed.xMax - (dx / plotWidth) * (zoomed.xMax - zoomed.xMin),
      yMin: zoomed.yMin - (dy / plotHeight) * (zoomed.yMax - zoomed.yMin),
      yMax: zoomed.yMax - (dy / plotHeight) * (zoomed.yMax - zoomed.yMin),
    });
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4 || Math.abs(factor - 1) > 0.02) {
      suppressNextClick = true;
    }
    return;
  }

  if (activePointers.size === 1 && panStart) {
    const point = chartPointFromEvent(e);
    if (!point) return;
    const dx = point.x - panStart.point.x;
    const dy = point.y - panStart.point.y;
    const start = panStart.viewport;
    viewport.value = clampViewport({
      xMin: start.xMin - (dx / plotWidth) * (start.xMax - start.xMin),
      xMax: start.xMax - (dx / plotWidth) * (start.xMax - start.xMin),
      yMin: start.yMin - (dy / plotHeight) * (start.yMax - start.yMin),
      yMax: start.yMax - (dy / plotHeight) * (start.yMax - start.yMin),
    });
    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) suppressNextClick = true;
  }
};

const onPointerEnd = (e: PointerEvent) => {
  const svg = e.currentTarget as SVGSVGElement;
  svg.releasePointerCapture?.(e.pointerId);
  activePointers.delete(e.pointerId);
  if (activePointers.size === 0) {
    panStart = null;
    pinchStart = null;
  } else if (activePointers.size === 1) {
    pinchStart = null;
    const [remaining] = [...activePointers.values()];
    const point = chartPointFromClient(svg, remaining.x, remaining.y);
    panStart = point ? { point, viewport: { ...viewport.value } } : null;
  }
};

// Build a 5-point star path centered at (cx, cy) with the given outer radius.
const starPath = (cx: number, cy: number, r: number): string => {
  const inner = r * 0.45;
  const pts: string[] = [];
  for (let i = 0; i < 10; i++) {
    const radius = i % 2 === 0 ? r : inner;
    // Top vertex first: rotate so the star points up.
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    pts.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return `M${pts[0]} L${pts.slice(1).join(" L")} Z`;
};

const hovered = ref<Point | null>(null);
const hoverPos = ref<{ x: number; y: number; flipX: boolean; flipY: boolean }>({
  x: 0,
  y: 0,
  flipX: false,
  flipY: false,
});

// On touch/coarse-pointer devices we show a minimal tooltip (model ID only),
// triggered by tap, and dismissed by tapping anywhere else.
const isCoarsePointer = ref(false);
let coarseMq: MediaQueryList | null = null;
const onCoarseChange = (e: MediaQueryListEvent) => {
  isCoarsePointer.value = e.matches;
};
const onDocPointerDown = (e: PointerEvent) => {
  if (!hovered.value) return;
  const target = e.target as Element | null;
  if (target && target.closest(".point")) return;
  hovered.value = null;
};

onMounted(() => {
  if (typeof window === "undefined" || !window.matchMedia) return;
  coarseMq = window.matchMedia("(pointer: coarse)");
  isCoarsePointer.value = coarseMq.matches;
  coarseMq.addEventListener?.("change", onCoarseChange);
  document.addEventListener("pointerdown", onDocPointerDown);
});

onBeforeUnmount(() => {
  coarseMq?.removeEventListener?.("change", onCoarseChange);
  if (typeof document !== "undefined") {
    document.removeEventListener("pointerdown", onDocPointerDown);
  }
});

const TOOLTIP_W = 280; // approximate, kept in sync with .tooltip max-width
const TOOLTIP_H = 240; // generous; the actual height depends on meta length

const positionFromEvent = (e: { clientX: number; clientY: number; currentTarget: EventTarget | null }, w: number, h: number) => {
  const target = e.currentTarget as SVGElement;
  const svg = target.ownerSVGElement as SVGSVGElement;
  const rect = svg.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  hoverPos.value = {
    x,
    y,
    flipX: x + 24 + w > rect.width,
    flipY: y + 24 + h > rect.height,
  };
};

const onMove = (e: MouseEvent, p: Point) => {
  if (isCoarsePointer.value) return;
  hovered.value = p;
  positionFromEvent(e, TOOLTIP_W, TOOLTIP_H);
};

const onLeave = () => {
  if (isCoarsePointer.value) return;
  hovered.value = null;
};

const onPointClick = (e: MouseEvent, p: Point) => {
  if (suppressNextClick) {
    suppressNextClick = false;
    return;
  }
  if (!isCoarsePointer.value) return;
  if (hovered.value === p) {
    hovered.value = null;
    return;
  }
  hovered.value = p;
  // Smaller tooltip on mobile — only the model line is shown.
  positionFromEvent(e, 200, 40);
};

const tooltipStyle = computed(() => {
  const x = hoverPos.value.x;
  const y = hoverPos.value.y;
  const w = isCoarsePointer.value ? 200 : TOOLTIP_W;
  const h = isCoarsePointer.value ? 40 : TOOLTIP_H;
  return {
    left: hoverPos.value.flipX ? `${Math.max(0, x - w - 12)}px` : `${x + 12}px`,
    top: hoverPos.value.flipY ? `${Math.max(0, y - h - 12)}px` : `${y + 12}px`,
  };
});
</script>

<template>
  <div class="scatter">
    <div class="scatter-toolbar">
      <span class="scatter-help">
        X-axis:
        {{ xAxis === "parameters" ? "Parameters (log)" : "Release date" }}.
        Y-axis: Rank score (lower is better).
        Showing {{ visiblePoints.length }} of {{ allPoints.length }} models with
        {{ xAxis === "parameters" ? "parameter counts" : "release dates" }}.
        Scroll or pinch to zoom; drag to pan.
      </span>
      <input
        v-model="searchQuery"
        type="search"
        class="scatter-search"
        placeholder="Search models..."
        aria-label="Search models"
      />
      <button
        v-if="isZoomed"
        type="button"
        class="pareto-toggle reset-zoom"
        aria-label="Reset scatter plot zoom"
        title="Reset zoom"
        @click="resetZoom"
      >
        Reset zoom
      </button>
      <button
        type="button"
        class="pareto-toggle"
        :class="{ active: showPareto }"
        :aria-pressed="showPareto"
        :aria-label="paretoToggleLabel"
        :title="paretoDescription"
        @click="togglePareto"
      >
        <svg viewBox="0 0 18 12" aria-hidden="true">
          <polyline points="1,10 6,7 10,7 17,1" />
        </svg>
        Pareto curve
      </button>
    </div>

    <div v-if="allPoints.length > 0" class="scatter-legend">
      <button
        v-for="k in presentKinds"
        :key="k"
        type="button"
        class="legend-item"
        :class="{ disabled: isKindHidden(k) }"
        @click="toggleKind(k)"
        :aria-pressed="!isKindHidden(k)"
      >
        <svg viewBox="0 0 14 14" class="swatch" aria-hidden="true">
          <circle cx="7" cy="7" r="5" :fill="KIND_COLOR[k]" />
        </svg>
        {{ KIND_LABEL[k] }}
      </button>
      <span class="legend-sep" v-if="hasCommercial || hasNonCommercial">|</span>
      <button
        v-if="hasCommercial"
        type="button"
        class="legend-item"
        :class="{ disabled: hideCommercial }"
        @click="hideCommercial = !hideCommercial"
        :aria-pressed="!hideCommercial"
      >
        <svg viewBox="0 0 14 14" class="swatch" aria-hidden="true">
          <path :d="starPath(7, 7, 6)" fill="currentColor" />
        </svg>
        Commercial
      </button>
      <button
        v-if="hasNonCommercial"
        type="button"
        class="legend-item"
        :class="{ disabled: hideNonCommercial }"
        @click="hideNonCommercial = !hideNonCommercial"
        :aria-pressed="!hideNonCommercial"
      >
        <svg viewBox="0 0 14 14" class="swatch" aria-hidden="true">
          <circle cx="7" cy="7" r="5" fill="currentColor" />
        </svg>
        Non-commercial
      </button>
    </div>

    <div v-if="allPoints.length === 0" class="scatter-empty" role="status">
      No ranked models have
      {{ xAxis === "parameters" ? "parameter counts" : "release dates" }}
      available yet.
    </div>

    <div v-else class="scatter-wrap">
      <svg
        :viewBox="`0 0 ${width} ${height}`"
        preserveAspectRatio="xMidYMid meet"
        class="scatter-svg"
        role="img"
        :aria-label="chartAriaLabel"
        @wheel.prevent="onWheel"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerEnd"
        @pointercancel="onPointerEnd"
      >
        <defs>
          <clipPath :id="clipId">
            <rect
              :x="margin.left"
              :y="margin.top"
              :width="plotWidth"
              :height="plotHeight"
            />
          </clipPath>
        </defs>

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
            :y1="yScale(t.value)"
            :y2="yScale(t.value)"
          />
          <text
            v-for="(t, i) in yTicks"
            :key="`yt-${i}`"
            class="tick-label"
            :x="margin.left - 8"
            :y="yScale(t.value)"
            dominant-baseline="middle"
            text-anchor="end"
          >
            {{ formatY(t.value) }}
          </text>
        </g>

        <!-- X ticks (major: solid grid, full label; minor: dashed, smaller label) -->
        <g class="grid">
          <line
            v-for="(t, i) in xTicks"
            :key="`xg-${i}`"
            :x1="xScale(t.value)"
            :x2="xScale(t.value)"
            :y1="margin.top"
            :y2="height - margin.bottom"
            :class="{ minor: !t.major }"
          />
          <text
            v-for="(t, i) in xTicks"
            :key="`xt-${i}`"
            :class="['tick-label', { minor: !t.major }]"
            :x="xScale(t.value)"
            :y="height - margin.bottom + 16"
            text-anchor="middle"
          >
            {{ formatX(t.value) }}
          </text>
        </g>

        <!-- Axis labels -->
        <text
          class="axis-label"
          :x="width / 2"
          :y="height - 6"
          text-anchor="middle"
        >
          {{ xAxis === "parameters" ? "Parameters" : "Release date" }}{{
            useLogX ? " (log)" : ""
          }}
        </text>
        <text
          class="axis-label"
          :transform="`translate(14, ${height / 2}) rotate(-90)`"
          text-anchor="middle"
        >
          Rank score
        </text>

        <g
          v-if="showPareto && paretoPolyline"
          class="pareto-frontier"
          :clip-path="`url(#${clipId})`"
          aria-hidden="true"
        >
          <polyline :points="paretoPolyline" />
        </g>

        <!-- Points: commercial → star, otherwise → circle. Colored by kind. -->
        <g class="points" :clip-path="`url(#${clipId})`">
          <template v-for="(p, i) in visiblePoints" :key="`p-${i}`">
            <path
              v-if="p.commercial"
              :d="starPath(xScale(p.x), yScale(p.y), 7)"
              :fill="KIND_COLOR[p.kind]"
              class="point"
              :class="{
                'pareto-point': showPareto && paretoPointSet.has(p),
              }"
              @mousemove="onMove($event, p)"
              @mouseleave="onLeave"
              @click.stop="onPointClick($event, p)"
            />
            <circle
              v-else
              :cx="xScale(p.x)"
              :cy="yScale(p.y)"
              r="5"
              :fill="KIND_COLOR[p.kind]"
              class="point"
              :class="{
                'pareto-point': showPareto && paretoPointSet.has(p),
              }"
              @mousemove="onMove($event, p)"
              @mouseleave="onLeave"
              @click.stop="onPointClick($event, p)"
            />
          </template>
        </g>
      </svg>

      <div
        v-if="hovered"
        class="tooltip"
        :class="{ compact: isCoarsePointer }"
        :style="tooltipStyle"
      >
        <div class="tt-model">{{ hovered.icon }} {{ hovered.label }}</div>
        <template v-if="!isCoarsePointer">
          <div class="tt-row">
            <span class="tt-label">Mean rank</span>
            <span class="tt-value">{{ hovered.y.toFixed(2) }}</span>
          </div>
          <div
            v-if="showPareto && paretoPointSet.has(hovered)"
            class="tt-row pareto-status"
          >
            <span class="tt-label">Pareto frontier</span>
            <span class="tt-value">Yes</span>
          </div>
          <div v-if="xAxis === 'releaseDate'" class="tt-row">
            <span class="tt-label">Release date</span>
            <span class="tt-value">{{ hovered.xLabel }}</span>
          </div>
          <div class="tt-row">
            <span class="tt-label">Kind</span>
            <span class="tt-value">{{ KIND_LABEL[hovered.kind] }}</span>
          </div>
          <div v-for="m in hovered.meta" :key="m.label" class="tt-row">
            <span class="tt-label">{{ m.label }}</span>
            <span class="tt-value">{{ m.value }}</span>
          </div>
        </template>
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

.scatter-search {
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-radius: 3px;
  padding: 0.2rem 0.4rem;
  font: inherit;
  font-size: 0.78rem;
  min-width: 180px;
}

.scatter-search:focus {
  outline: 1px solid var(--color-link);
  outline-offset: -1px;
}

.scatter-help {
  flex: 1 1 300px;
  font-size: 0.8rem;
}

.pareto-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  background: var(--color-bg);
  color: var(--color-muted);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.2rem 0.5rem;
  font: inherit;
  font-size: 0.78rem;
  white-space: nowrap;
  cursor: pointer;
}

.pareto-toggle:hover {
  color: var(--color-text);
  border-color: var(--color-muted);
}

.pareto-toggle.active {
  background: var(--color-surface);
  color: var(--color-link);
  border-color: var(--color-link);
}

.pareto-toggle:focus-visible {
  outline: 2px solid var(--color-link);
  outline-offset: 2px;
}

.pareto-toggle svg {
  width: 18px;
  height: 12px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.scatter-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.9rem;
  font-size: 0.8rem;
  color: var(--color-muted);
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  background: transparent;
  border: 0;
  color: inherit;
  font: inherit;
  font-size: 0.8rem;
  cursor: pointer;
  padding: 0.1rem 0.2rem;
  border-radius: 3px;
  transition: opacity 0.15s ease;
}

.legend-item:hover {
  background: var(--color-surface);
}

.legend-item.disabled {
  opacity: 0.35;
  text-decoration: line-through;
}

.legend-sep {
  opacity: 0.4;
}

.swatch {
  width: 14px;
  height: 14px;
  display: inline-block;
}

.scatter-wrap {
  position: relative;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 0.5rem;
}

.scatter-empty {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  color: var(--color-muted);
  padding: 2rem 1rem;
  text-align: center;
}

.scatter-svg {
  width: 100%;
  height: auto;
  display: block;
  cursor: grab;
  touch-action: none;
  user-select: none;
}

.axis {
  stroke: var(--color-border);
  stroke-width: 1;
}

.grid line {
  stroke: var(--color-muted);
  stroke-dasharray: 3 3;
  opacity: 0.55;
}

.grid line.minor {
  opacity: 0.3;
  stroke-dasharray: 2 5;
}

.tick-label,
.axis-label {
  fill: var(--color-muted);
  font-size: 11px;
}

.tick-label.minor {
  font-size: 9px;
  opacity: 0.7;
}

.pareto-frontier {
  pointer-events: none;
}

.pareto-frontier polyline {
  fill: none;
  stroke: var(--color-text);
  stroke-width: 2.25;
  stroke-dasharray: 6 4;
  stroke-linecap: round;
  stroke-linejoin: round;
  opacity: 0.9;
}

.point {
  fill-opacity: 0.85;
  stroke: var(--color-bg);
  stroke-width: 1;
  cursor: pointer;
  transition:
    fill-opacity 0.1s ease,
    stroke-width 0.1s ease;
}

.point.pareto-point {
  stroke: var(--color-text);
  stroke-width: 2.5;
}

.point:hover {
  fill-opacity: 1;
  stroke-width: 1.8;
}

.point.pareto-point:hover {
  stroke-width: 3.2;
}

.pareto-status .tt-label,
.pareto-status .tt-value {
  color: var(--color-text);
  font-weight: 600;
}

.tooltip {
  position: absolute;
  pointer-events: none;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 0.5rem 0.7rem;
  font-size: 0.78rem;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  width: 280px;
  max-width: calc(100% - 24px);
  z-index: 5;
}

.tooltip.compact {
  width: auto;
  max-width: calc(100% - 24px);
  padding: 0.35rem 0.55rem;
}

.tooltip.compact .tt-model {
  margin-bottom: 0;
  border-bottom: 0;
  padding-bottom: 0;
  white-space: normal;
  overflow-wrap: anywhere;
}

.tt-model {
  font-weight: 500;
  margin-bottom: 0.35rem;
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 0.25rem;
  white-space: nowrap;
}

.tt-row {
  display: flex;
  justify-content: space-between;
  gap: 1.25rem;
  line-height: 1.4;
}

.tt-label {
  color: var(--color-muted);
}

.tt-value {
  font-variant-numeric: tabular-nums;
}
</style>
