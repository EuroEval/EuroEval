import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/docs/README.md" },
    { path: "/docs/:path", name: "doc" },
    { path: "/docs/:path/", name: "doc-trailing" },
  ],
});

export default router;
