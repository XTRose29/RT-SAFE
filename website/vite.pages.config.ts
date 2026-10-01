import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  base: "./",
  plugins: [react()],
  define: {
    __RT_SAFE_REPOSITORY_URL__: JSON.stringify(process.env.REPOSITORY_URL || ""),
  },
  build: { outDir: "dist-pages", emptyOutDir: true },
  server: { port: 4173, host: "0.0.0.0" },
  preview: { port: 4173 },
});
