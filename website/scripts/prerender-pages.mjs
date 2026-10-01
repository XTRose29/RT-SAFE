import { createServer } from "vite";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: root + "/vite.pages.config.ts",
  server: { middlewareMode: true },
  appType: "custom",
});
try {
  const { render } = await server.ssrLoadModule("/pages-ssr.tsx");
  const path = root + "/dist-pages/index.html";
  const html = readFileSync(path, "utf8").replace(
    '<div id="root"></div>',
    `<div id="root">${render()}</div>`,
  );
  writeFileSync(path, html);
  writeFileSync(root + "/dist-pages/.nojekyll", "");
  console.log(
    "Pre-rendered full page content for fast initial display and search indexing.",
  );
} finally {
  await server.close();
}
