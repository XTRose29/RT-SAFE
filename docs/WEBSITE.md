# Local website and GitHub Pages

The website is a pre-rendered React/Vite static site. No server, database, secret, or API key is needed at deployment time.

Project URL: **https://xtrose29.github.io/RT-SAFE/**.
Repository: **https://github.com/XTRose29/RT-SAFE**.
Updates to `website/` on `main` automatically build and deploy through GitHub Actions.

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

1. Authenticate using `gh auth login`, then run `./scripts/create-github-repo.sh` from the repository root. It creates a private `RT-SAFE` repository under your account and pushes the committed files. Pass `OWNER/NAME` to choose a different destination.
2. When ready to publish, make the repository public using Settings → General → Change repository visibility. This publishes the benchmark source, website, and included media.
3. In repository Settings → Pages, select **GitHub Actions** as the build source.
4. Set the repository Actions variable `PAGES_ENABLED` to `true`.
5. Run the **Deploy project website** workflow, or push to `main`.

The workflow runs type checks, data tests and the production build, sets canonical/share URLs, then deploys the static artifact. CI also builds the site while the repository is private, without publishing it.

For an ordinary repository named `RT-SAFE`, the URL is `https://OWNER.github.io/RT-SAFE/`. For the root address `https://rt-safe.github.io/`, you must control the `rt-safe` GitHub account or organization and create a repository named `rt-safe.github.io` there. The same build supports both URL shapes. Availability of private-repository Pages depends on the account plan; publication should occur after the public release is ready.

Set repository variable `SITE_URL` only when using a custom domain. Configure that domain in Pages settings and its DNS separately.

## Content and film

- `website/app/rtsafe.tsx`: page and interactive tables.
- `website/app/data.json`: manuscript results.
- `website/app/examples.json`: curated recorded decisions.
- `website/public/media/rt-safe-video.mp4`: canonical 90-second film without audio.
- `website/public/media/rt-safe-nyc-90s.mp4`: identical compatibility copy for existing links.
- `website/public/media/logos/simworld.png`: supplied SimWorld emblem used in the website brand suite and film opening.
- `website/public/media/rt-safe-nyc-90s-captioned.mp4`: legacy URL retained as a silent, subtitle-free copy.
- `website/public/media/rt-safe-original-comparison.mp4`: current Astra/Sol comparison using original recorded snapshots at 6× simulation speed.
- `website/public/media/rt-safe-nyc-comparison.mp4`: earlier NYC reconstruction, retained as an alternate.
- `website/public/media/nyc/`: native city footage, synchronized timing/replay clips, and camera gallery.
- `website/nyc/`: native rendering scripts and curated reconstruction evidence.
- `website/video/nyc-90s/`: timing, captions, and export validation.

[NYC production instructions](../website/nyc/README.md) describe the isolated Unreal project and Movie Render Queue workflow. [Film instructions](../website/video/nyc-90s/README.md) describe editing and export. The timing encounter and environment tour are illustrative. The matched Astra/Sol replay visualizes an existing recorded run; manuscript results remain from the original benchmark maps.

Do not add raw research folders or environment files to `website/public`: everything there is deployed publicly.

[GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

The current revision adds a visible stop and recoil to the illustrative park contact, full-stride walking, an explicit RT-Safe introduction, larger opening headings, and animated leaderboard/radar results. Original replay snapshots use brief dissolves; collision report times and all paper measurements are unchanged.


## Interactive NYC environment

The environment section uses four native 360° cubemap captures: the approach, the path,
the crossing, and Madison Square Park. Drag, touch, or use arrow keys to look; use the
viewpoint buttons to move between capture positions. Camera rotation can be played and
paused. The moving-scene option uses the original 12-second Unreal rendering without
presentation overlays. These are captured views, not a live benchmark simulation.

All public MP4 files contain video only. Website players have no caption tracks, and
older `captioned` URLs now serve the matching uncaptioned video for compatibility.
Run `website/scripts/prepare-silent-web-media.py` after regenerating legacy exports.
The supplied `website/scripts/verify-environment-explorer.cjs` checks the browser controls.
