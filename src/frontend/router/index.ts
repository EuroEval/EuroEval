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
  scrollBehavior: () => ({ top: 0 }),
});

export default router;
