import { readFileSync, writeFileSync } from "node:fs";
const value = process.env.SITE_URL || process.env.PAGES_URL;
if (!value)
  throw new Error("Set SITE_URL or PAGES_URL to your final website URL.");
let url = new URL(value);
if (
  url.hostname.endsWith(".github.io") &&
  url.pathname.toLowerCase() === `/${url.hostname}/`
)
  url = new URL(url.origin + "/");
if (!url.pathname.endsWith("/")) url.pathname += "/";
const htmlPath = new URL("../dist-pages/index.html", import.meta.url);
let html = readFileSync(htmlPath, "utf8");
html = html.replace(/<link rel="canonical"[^>]*\/?>/g, "");
html = html.replace(/<meta property="og:url"[^>]*\/?>/g, "");
html = html.replace(
  /(<meta (?:property="og:image"|name="twitter:image") content=")[^"]+"/g,
  `$1${new URL("og.png", url)}"`,
);
html = html.replace(
  "</head>",
  `<link rel="canonical" href="${url}"/><meta property="og:url" content="${url}"/></head>`,
);
writeFileSync(htmlPath, html);
writeFileSync(new URL("../dist-pages/.nojekyll", import.meta.url), "");
console.log("Sharing metadata configured for", url.toString());
