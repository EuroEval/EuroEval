import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/documentation/README.md" },
    { path: "/documentation/:path", name: "doc" },
    { path: "/documentation/:path/", name: "doc-trailing" },
  ],
});

export default router;
