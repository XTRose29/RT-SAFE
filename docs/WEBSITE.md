# Local website and GitHub Pages

The website is a pre-rendered React/Vite static site. No server, database, secret, or API key is needed at deployment time.

## Local preview

Use Node.js 22.13 or newer:

```bash
cd website
npm ci
npm run typecheck
npm test
npm run build
npm run preview -- --host 127.0.0.1
```

Open the address printed by Vite. `npm run dev` starts the editing server. The deployable folder is `website/dist-pages/`.

## Publish from this repository

1. Push the repository to GitHub.
2. In repository Settings → Pages, select **GitHub Actions** as the build source.
3. Set the repository Actions variable `PAGES_ENABLED` to `true`.
4. Run the **Deploy project website** workflow, or push to `main`.

The workflow runs type checks, data tests and the production build, sets canonical/share URLs, then deploys the static artifact. CI also builds the site while the repository is private, without publishing it.

For an ordinary repository named `RT-SAFE`, the URL is `https://OWNER.github.io/RT-SAFE/`. For the root address `https://rt-safe.github.io/`, you must control the `rt-safe` GitHub account or organization and create a repository named `rt-safe.github.io` there. The same build supports both URL shapes. Availability of private-repository Pages depends on the account plan; publication should occur after the public release is ready.

Set repository variable `SITE_URL` only when using a custom domain. Configure that domain in Pages settings and its DNS separately.

## Content and film

- `website/app/rtsafe.tsx`: page and interactive tables.
- `website/app/data.json`: manuscript results.
- `website/app/examples.json`: curated recorded decisions.
- `website/public/media/rt-safe-demo.mp4`: full narrated film.
- `website/public/media/rt-safe-highlights.mp4`: one-minute cut.
- `website/public/media/rt-safe-conference-captioned.mp4`: full film with burned captions.
- `website/video/`: script, timing, voice clips and source footage.

[Video editing instructions](../website/video/README.md) describe regeneration. The concept image is labeled as illustrative; the decision examples use recorded UE observations.

Do not add raw research folders or environment files to `website/public`: everything there is deployed publicly.

[GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).
