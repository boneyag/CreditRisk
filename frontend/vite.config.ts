import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiProxyTarget = process.env.API_PROXY_TARGET?.trim() || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/explain": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      "/predict": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      "/health": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
      "/model_info": {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
});
