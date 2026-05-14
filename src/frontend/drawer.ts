import { ref } from "vue";

export const drawerOpen = ref(false);

export const openDrawer = () => {
  drawerOpen.value = true;
};

export const closeDrawer = () => {
  drawerOpen.value = false;
};

export const toggleDrawer = () => {
  drawerOpen.value = !drawerOpen.value;
};
