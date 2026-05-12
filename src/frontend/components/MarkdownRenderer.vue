<script setup lang="ts">
import MarkdownIt from "markdown-it";
import hljs from "highlight.js";
import { ref, watch } from "vue";

const props = defineProps<{
  path: string;
}>();

const docTitle = () => {
  const name = props.path.replace(/\.(md|mdx|txt)$/i, "");
  return name.split("/").pop() || name;
};

const breadcrumb = () => {
  const parts = props.path.split("/");
  return parts.slice(0, -1).join("/");
};

const html = ref("");
const loading = ref(true);
const error = ref(false);

const md = new MarkdownIt({
  html: true,
  linkify: true,
  highlight: (code, lang) => {
    const language = lang?.replace(/\W/g, "") || "plaintext";
    const escaped = hljs.highlight(code, {
      language,
      ignoreDelinser: true,
    }).value;
    return `<pre class="code-block"><code class="hljs language-${language}">${escaped}</code></pre>`;
  },
});

watch(
  () => props.path,
  async (newPath) => {
    loading.value = true;
    error.value = false;
    html.value = "";

    try {
      const response = await fetch(`/docs/${newPath}`);
      if (!response.ok) {
        error.value = true;
        loading.value = false;
        return;
      }
      const text = await response.text();
      html.value = md.render(text);
      loading.value = false;
    } catch {
      error.value = true;
      loading.value = false;
    }
  },
  { immediate: true }
);
</script>

<template>
  <div v-if="loading" class="loading">Loading...</div>
  <div v-else-if="error" class="error">Failed to load content.</div>
  <div v-else class="page-wrapper">
    <div class="page-header">
      <div class="breadcrumb">
        <a href="/docs">docs</a>
        <span v-if="breadcrumb" class="separator">/</span>
        <span v-if="breadcrumb" class="breadcrumb-path">{{ breadcrumb }}</span>
      </div>
      <h2 class="page-title">{{ docTitle }}</h2>
    </div>
    <div class="markdown-body" v-html="html" />
  </div>
</template>

<style scoped>
.markdown-body {
  font-family: sans-serif;
  line-height: 1.6;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3 {
  margin-top: 1.5rem;
  margin-bottom: 0.5rem;
}

.markdown-body code {
  background: #f5f5f5;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
}

.markdown-body pre {
  background: #f5f5f5;
  padding: 1rem;
  overflow-x: auto;
  border-radius: 4px;
}

.markdown-body blockquote {
  border-left: 4px solid #ccc;
  padding-left: 1rem;
  color: #666;
}
</style>
