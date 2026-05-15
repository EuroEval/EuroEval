import { createRouter, createWebHistory } from "vue-router";
import { defineComponent, h } from "vue";

const PageStub = defineComponent({
  render: () => h("div"),
});

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "home", component: PageStub },
    { path: "/:section", name: "section", component: PageStub },
    { path: "/:section/:page", name: "section-page", component: PageStub },
  ],
  scrollBehavior: (to) => {
    // If the target has a hash, let the markdown components scroll to it
    // after they've expanded any <details> wrappers. Otherwise jump to the
    // top of the page on every navigation.
    if (to.hash) return false;
    return { top: 0 };
  },
});

export default router;
