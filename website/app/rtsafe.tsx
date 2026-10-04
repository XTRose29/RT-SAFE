"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import rawData from "./data.json";
import { BehaviorProfiles } from "./behavior-profiles";
import { NYCHero, NYCTiming, NYCComparison, NYCEnvironment, NYCSceneGallery } from "./nyc-scenes";

declare const __RT_SAFE_REPOSITORY_URL__: string;
const repositoryUrl = typeof __RT_SAFE_REPOSITORY_URL__ !== "undefined" ? __RT_SAFE_REPOSITORY_URL__ : "";

type Scope = "realtime" | "static" | "average";
type Metric = {
  success: number;
  safeSuccess: number;
  spl: number;
  collisions: number;
  latency: number;
  decisions: number;
  active?: number;
  passive?: number;
  passiveShare?: number;
};
type Model = {
  name: string;
  fullName: string;
  provider: string;
  color: string;
  realtime: Metric;
  static: Metric;
  average: Metric;
  effort: {
    label: string;
    collisions: number;
    latency: number;
    active: number;
    passive: number;
  }[];
};
const models = rawData.models as Model[];
const fmt = (n: number | undefined, d = 1) =>
  n === undefined ? "—" : n.toFixed(d);
const Arrow = ({ diagonal = false }: { diagonal?: boolean }) => (
  <span aria-hidden="true">{diagonal ? "↗" : "↘"}</span>
);
function SectionHead({
  n,
  label,
  title,
  description,
}: {
  n: string;
  label: string;
  title: React.ReactNode;
  description?: string;
}) {
  return (
    <div className="section-head">
      <div className="eyebrow">
        <span className="section-number">{n}</span>
        {label}
      </div>
      <div className="section-title">
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
    </div>
  );
}
function Results() {
  const [scope, setScope] = useState<Scope>("average");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<keyof Metric>("collisions");
  const [asc, setAsc] = useState(true);
  const [selected, setSelected] = useState("Fable");
  const list = useMemo(
    () =>
      models
        .filter((m) =>
          (m.fullName + " " + m.provider)
            .toLowerCase()
            .includes(search.toLowerCase()),
        )
        .sort(
          (a, b) =>
            ((a[scope][sort] ?? 0) - (b[scope][sort] ?? 0)) * (asc ? 1 : -1),
        ),
    [scope, search, sort, asc],
  );
  const model = models.find((m) => m.name === selected)!;
  const score = model[scope];
  const changeSort = (key: keyof Metric) => {
    if (key === sort) setAsc(!asc);
    else {
      setSort(key);
      setAsc(!["success", "safeSuccess", "spl"].includes(key));
    }
  };
  const cols: [keyof Metric, string, string][] = [
    ["success", "Success", "↑"],
    ["safeSuccess", "Safe success", "↑"],
    ["collisions", "Collisions / ep.", "↓"],
    ["spl", "SPL", "↑"],
    ["latency", "Latency", "s"],
    ["decisions", "Decisions / ep.", ""],
  ];
  return (
    <div className="results-explorer">
      <div className="results-toolbar">
        <div className="segmented" aria-label="Result condition">
          {(
            [
              ["realtime", "Hard · Real-time"],
              ["static", "Hard · Static"],
              ["average", "All difficulties"],
            ] as [Scope, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              aria-pressed={scope === key}
              onClick={() => setScope(key)}
            >
              {label}
            </button>
          ))}
        </div>
        <label className="search">
          <span aria-hidden="true">⌕</span>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Find a model…"
            aria-label="Filter models"
          />
        </label>
        <a className="text-link csv-link" href="data/results.csv" download>
          CSV <span>↓</span>
        </a>
      </div>
      <div className="results-context">
        <span>
          <i className="live-dot" />
          {scope === "average"
            ? "108 episodes per model · Easy, medium & hard · Real-time"
            : "36 matched routes per model · Hard · Provider-default reasoning"}
        </span>
        <span>{list.length} models</span>
      </div>
      <div className="results-workspace">
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Scrollable model results"
        >
          <table>
            <caption className="sr-only">
              RT-SAFE results, {scope}. Select a model to inspect its collision
              counts. Select column headers to sort.
            </caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                {cols.map(([key, label, direction]) => (
                  <th
                    scope="col"
                    key={key}
                    aria-sort={
                      sort === key ? (asc ? "ascending" : "descending") : "none"
                    }
                  >
                    <button onClick={() => changeSort(key)}>
                      {label}
                      <span>
                        {sort === key ? (asc ? "↑" : "↓") : direction}
                      </span>
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {list.map((m) => (
                <tr
                  key={m.name}
                  className={m.name === selected ? "selected" : ""}
                >
                  <th scope="row">
                    <button
                      onClick={() => setSelected(m.name)}
                      aria-pressed={m.name === selected}
                    >
                      <i style={{ background: m.color }} />
                      <span>
                        {m.name}
                        <small>{m.provider}</small>
                      </span>
                    </button>
                  </th>
                  {cols.map(([key]) => (
                    <td
                      key={key}
                      className={
                        key === "collisions"
                          ? "collision-cell"
                          : key === "safeSuccess"
                            ? "safe-cell"
                            : ""
                      }
                    >
                      {fmt(
                        m[scope][key],
                        key === "spl" ? 3 : key === "collisions" ? 2 : 1,
                      )}
                      {["success", "safeSuccess"].includes(key) && (
                        <small>%</small>
                      )}
                      {key === "latency" && <small> s</small>}
                    </td>
                  ))}
                </tr>
              ))}
              {!list.length && (
                <tr>
                  <td colSpan={7} className="empty">
                    No matching models.{" "}
                    <button onClick={() => setSearch("")}>Clear search</button>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <aside className="model-inspector" aria-live="polite">
          <span className="micro">MODEL IN FOCUS</span>
          <h3>{model.name}</h3>
          <p>{model.fullName}</p>
          <div className="model-score">
            <strong>{fmt(score.collisions, 2)}</strong>
            <span>collisions / episode</span>
          </div>
          {scope !== "average" ? (
            <>
              <div
                className="stacked-bar"
                aria-label={`${score.active} active and ${score.passive} passive collisions per episode`}
              >
                <span
                  style={{
                    width: `${((score.active ?? 0) / score.collisions) * 100}%`,
                  }}
                />
                <i style={{ flex: 1 }} />
              </div>
              <dl className="collision-legend">
                <div>
                  <dt>
                    <i className="active-color" />
                    While acting
                  </dt>
                  <dd>{fmt(score.active, 2)}</dd>
                </div>
                <div>
                  <dt>
                    <i className="passive-color" />
                    While reasoning
                  </dt>
                  <dd>{fmt(score.passive, 2)}</dd>
                </div>
              </dl>
              <p className="inspector-note">
                {scope === "static"
                  ? "The world is paused during inference. Passive contacts are zero."
                  : `${Math.round(((score.passive ?? 0) / score.collisions) * 100)}% of this model’s contacts occur during inference.`}
              </p>
            </>
          ) : (
            <p className="inspector-note">
              {score.passiveShare}% of contacts occur during inference, across
              the three difficulties.
            </p>
          )}
          <div className="model-compare">
            <span>Static → Real-time</span>
            <strong>
              {(model.realtime.collisions / model.static.collisions).toFixed(1)}
              ×
            </strong>
            <small>change in collisions, hard setting</small>
          </div>
        </aside>
      </div>
      <div className="table-note">
        <b>Read the metrics.</b> Success means arrival. Safe success also
        requires zero collisions, hazard interactions, and traffic violations.
        SPL measures successful path efficiency. Lower collision counts are
        better. Counts include successful and failed episodes.
      </div>
    </div>
  );
}
function Reasoning() {
  const [name, setName] = useState("Fable");
  const m = models.find((x) => x.name === name)!;
  return (
    <div className="reasoning-panel">
      <div className="reasoning-copy">
        <div className="eyebrow">RQ3 / THE REASONING TRADEOFF</div>
        <h3>
          More thinking.
          <br />
          More time exposed.
        </h3>
        <p>
          Increasing effort above provider defaults raises mean collisions from{" "}
          <b>40.7 to 62.0</b> per episode. Better individual actions can coexist
          with more contacts during inference.
        </p>
        <label className="select-label">
          Inspect a model
          <select value={name} onChange={(e) => setName(e.target.value)}>
            {models.map((x) => (
              <option key={x.name}>{x.name}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="effort-chart">
        <div className="chart-header">
          <span>COLLISIONS / EPISODE</span>
          <span>Hard · Real-time</span>
        </div>
        <div className="effort-bars">
          {m.effort.map((e, i) => (
            <div className="effort-column" key={e.label}>
              <span>{fmt(e.collisions)}</span>
              <div className="effort-track">
                <div style={{ height: `${(e.collisions / 100) * 100}%` }}>
                  <i
                    style={{ height: `${(e.passive / e.collisions) * 100}%` }}
                  />
                  <b style={{ flex: 1 }} />
                </div>
              </div>
              <strong>{["Lower", "Medium", "Higher"][i]}</strong>
              <small>{fmt(e.latency)} s response</small>
            </div>
          ))}
        </div>
        <div className="chart-legend">
          <span>
            <i className="active-color" />
            During action
          </span>
          <span>
            <i className="passive-color" />
            During inference
          </span>
        </div>
        <p className="fineprint">
          Effort levels are provider-specific. Sol’s default is Lower; the
          others default to Medium. Bar heights use the same 0–100 scale for
          every model.
        </p>
      </div>
    </div>
  );
}
function Findings() {
  return (
    <div className="findings">
      <div className="finding-intro">
        <span className="eyebrow">RQ2 / THE HIDDEN SAFETY GAP</span>
        <h3>
          High completion.
          <br />
          <em>Almost no safe arrivals.</em>
        </h3>
        <p>
          288 matched model–route pairs. The same initial configurations. A
          world that either pauses or keeps moving during inference.
        </p>
        <div className="finding-callout">
          <b>12.3×</b>
          <span>
            more collisions
            <br />
            in real time
          </span>
        </div>
      </div>
      <div className="comparison-chart">
        <div className="chart-legend">
          <span>
            <i className="static-color" />
            Static
          </span>
          <span>
            <i className="realtime-color" />
            Real-time
          </span>
        </div>
        {[
          { label: "Task completion", a: 91.3, b: 94.1 },
          { label: "Safe completion", a: 19.8, b: 0.7 },
        ].map((x) => (
          <div className="comparison-group" key={x.label}>
            <h4>
              {x.label}
              <span>episodes %</span>
            </h4>
            <div className="comparison-row">
              <span>Static</span>
              <div>
                <i className="static-bar" style={{ width: `${x.a}%` }} />
              </div>
              <b>{x.a}%</b>
            </div>
            <div className="comparison-row">
              <span>Real-time</span>
              <div>
                <i className="realtime-bar" style={{ width: `${x.b}%` }} />
              </div>
              <b>{x.b}%</b>
            </div>
          </div>
        ))}
        <p className="fineprint">
          Hard environments · Eight models · Provider-default reasoning.
          <br />
          Modes also use different timing instructions in their prompts.
        </p>
      </div>
    </div>
  );
}
function Training() {
  return (
    <div className="training">
      <div>
        <div className="eyebrow">FROM EVALUATION TO LEARNING</div>
        <h3>
          A testbed for
          <br />
          safer policies.
        </h3>
        <p>
          RT–SAFE also supplies progress rewards and safety costs for offline
          reinforcement learning. Reward design changes the balance between
          arrival, collision avoidance, and route progress.
        </p>
        <a
          className="text-link"
          href="media/rt-safe-paper.pdf"
          target="_blank"
          rel="noreferrer"
        >
          Read the training study <Arrow diagonal />
        </a>
      </div>
      <div className="training-results">
        <div className="chart-header">
          <span>HELD-OUT MAPS · 16 TASKS</span>
          <span>Qwen3-VL-4B</span>
        </div>
        <table>
          <caption className="sr-only">Offline training evaluation</caption>
          <thead>
            <tr>
              <th>Method</th>
              <th>Success ↑</th>
              <th>Coll. / 100 m ↓</th>
            </tr>
          </thead>
          <tbody>
            {rawData.rl.map((r) => (
              <tr key={r.method}>
                <th>
                  {r.method === "RL (wander penalty)"
                    ? "RL · wander penalty"
                    : r.method === "RL (base reward)"
                      ? "RL · base reward"
                      : r.method}
                </th>
                <td>{r.success.toFixed(1)}%</td>
                <td>{r.per100m.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="fineprint">
          Separate training study with a fixed 3 s decision delay. Behavior
          cloning has the lowest collision rate; the wander-penalty policy has
          the highest completion. This is a different protocol from the
          eight-model comparison.
        </p>
      </div>
    </div>
  );
}
export default function RTsafe() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [menu, setMenu] = useState(false);
  const [film, setFilm] = useState<"nyc-90s" | "nyc-comparison">("nyc-90s");
  const watch = () => {
    document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" });
    videoRef.current?.play().catch(() => {});
  };
  return (
    <>
      <a className="skip-link" href="#benchmark">
        Skip to benchmark
      </a>
      <header className="header">
        <a className="brand" href="#top" aria-label="RT-SAFE home">
          <span className="brand-icon">rt</span> RT–SAFE
          <span className="brand-note">The real-time safety benchmark</span>
        </a>
        <button
          className="menu-toggle"
          onClick={() => setMenu(!menu)}
          aria-expanded={menu}
          aria-controls="site-nav"
        >
          {menu ? "Close" : "Menu"} <span>{menu ? "−" : "+"}</span>
        </button>
        <nav id="site-nav" className={menu ? "open" : ""}>
          <a href="#benchmark" onClick={() => setMenu(false)}>
            Benchmark
          </a>
          <a href="#results" onClick={() => setMenu(false)}>
            Results
          </a>
          <a href="#resources" onClick={() => setMenu(false)}>
            Resources
          </a>
          <button
            onClick={() => {
              setMenu(false);
              watch();
            }}
          >
            Watch film <Arrow diagonal />
          </button>
        </nav>
      </header>
      <main id="top">
        <section className="hero">
          <div className="eyebrow">
            <i /> EMBODIED AI · REAL-TIME SAFETY
          </div>
          <h1>
            The world doesn’t pause
            <br />
            while an agent <em>thinks.</em>
          </h1>
          <div className="hero-bottom">
            <p>
              Neither should its safety evaluation. RT–SAFE measures how
              embodied agents navigate a world that keeps moving during
              reasoning and action.
            </p>
            <div className="hero-actions">
              <button className="light-button" onClick={watch}>
                <span className="play-icon">▶</span> Watch the film
              </button>
              <a className="button" href="#results">
                Explore the results <Arrow />
              </a>
            </div>
          </div>
          <div className="hero-scene">
            <NYCHero />
            <div className="scene-top">
              <span>
                <i /> THE WORLD IS STILL MOVING
              </span>
              <span>RT–SAFE / EMBODIED SAFETY</span>
            </div>
            <div className="scene-title">
              A decision takes seconds.
              <br />
              <span>The world only needs one.</span>
            </div>
            <button className="scene-preview" onClick={watch}>
              <img
                src="media/nyc/high_follow.webp"
                alt="Elevated third-person view in the NYC scene"
                width="720"
                height="640"
              />
              <span className="scene-play">▶</span>
              <span>
                Inside the benchmark <Arrow diagonal />
              </span>
            </button>
            <div className="scene-bottom">
              <span>Observe → Reason → Act</span>
              <span>Madison Square Park · Unreal Engine</span>
            </div>
          </div>
        </section>
        <NYCSceneGallery />
        <section className="intro-stats" aria-label="Benchmark scale">
          <div>
            <b>5</b>
            <span>city maps</span>
          </div>
          <div>
            <b>36</b>
            <span>navigation routes</span>
          </div>
          <div>
            <b>8</b>
            <span>vision-language models</span>
          </div>
          <div>
            <b>16</b>
            <span>available actions</span>
          </div>
        </section>
        <section id="benchmark" className="section">
          <SectionHead
            n="01"
            label="WHY REAL TIME MATTERS"
            title={
              <>
                A safe action can
                <br />
                arrive too late.
              </>
            }
            description="Pedestrians move. Vehicles approach. Signals change. RT–SAFE evaluates both the decision an agent makes and the world in which that decision finally executes."
          />
          <NYCTiming />
          <div className="principles">
            <div>
              <span>01 / OBSERVE</span>
              <h3>A snapshot of a moving world.</h3>
              <p>
                Recent motion frames and seven marked targets give the agent
                visual context for its next decision.
              </p>
            </div>
            <div>
              <span>02 / REASON</span>
              <h3>The clock keeps running.</h3>
              <p>
                The simulator evolves throughout planning. Contacts recorded in
                this interval are called passive collisions.
              </p>
            </div>
            <div>
              <span>03 / ACT</span>
              <h3>The outcome is measured.</h3>
              <p>
                The selected action executes. Active collisions, hazard
                interactions, and traffic violations enter the record.
              </p>
            </div>
          </div>
        </section>
        <section className="section safety-section">
          <SectionHead
            n="02"
            label="WHAT WE EVALUATE"
            title={
              <>
                One goal.
                <br />
                Three dimensions of safety.
              </>
            }
            description="Follow sidewalk and crosswalk subgoals to a destination. Reaching it safely means avoiding every recorded safety event along the way."
          />
          <NYCEnvironment />
          <div className="safety-cards">
            <article>
              <div className="safety-symbol collision-symbol">
                <i />
                <b />
                <span>↔</span>
              </div>
              <span className="micro">01 / CONTACT</span>
              <h3>Collision avoidance</h3>
              <p>
                People, moving objects, buildings, and vehicles. Events are
                separated into inference and action intervals.
              </p>
              <div className="card-tags">
                <span>Active</span>
                <span>Passive</span>
                <span>Vehicle impact</span>
              </div>
            </article>
            <article>
              <div className="safety-symbol hazard-symbol">
                <i />
                <b />
                <span>!</span>
              </div>
              <span className="micro">02 / TERRAIN</span>
              <h3>Hazard avoidance</h3>
              <p>
                Trip, oil, and water interactions test navigation around unsafe
                surfaces and motion disruptions.
              </p>
              <div className="card-tags">
                <span>Trip</span>
                <span>Oil</span>
                <span>Water</span>
              </div>
            </article>
            <article>
              <div className="safety-symbol signal-symbol">
                <i />
                <b />
                <span />
              </div>
              <span className="micro">03 / RULES</span>
              <h3>Traffic compliance</h3>
              <p>
                Use marked crossings and enter on WALK. Illegal entries trigger
                a controlled conflict vehicle.
              </p>
              <div className="card-tags">
                <span>Road entry</span>
                <span>Crossing signals</span>
              </div>
            </article>
          </div>
          <div className="safe-definition">
            <span className="checkmark">✓</span>
            <strong>Safe success</strong>
            <span>
              Reach the goal <b>+</b> zero collisions <b>+</b> zero hazard
              interactions <b>+</b> zero traffic violations
            </span>
          </div>
        </section>
        <section id="examples" className="section">
          <SectionHead
            n="03"
            label="RECORDED MODEL BEHAVIOR"
            title={<>Same route. Different decisions.</>}
            description="Follow GPT-6 Astra and GPT-5.6 Sol through the same final route segment. Their recorded decisions and timing are reconstructed in the NYC scene."
          />
          <NYCComparison />
          <div className="action-space">
            <span className="micro">16 ACTIONS, ONE SHARED INTERFACE</span>
            <div>
              <strong>7</strong>
              <span>
                Moves<small>1 m, 2 m, or 4 m</small>
              </span>
            </div>
            <div>
              <strong>6</strong>
              <span>
                Turns<small>±30°, ±60°, ±90°</small>
              </span>
            </div>
            <div>
              <strong>3</strong>
              <span>
                Waits<small>1, 2, or 3 seconds</small>
              </span>
            </div>
          </div>
        </section>
        <section id="results" className="section results-section">
          <SectionHead
            n="04"
            label="THE RESULTS"
            title={
              <>
                Arrival is only
                <br />
                part of the story.
              </>
            }
            description="Compare completion, safety, and decision behavior. Results are transcribed directly from the accompanying manuscript, with each evaluation condition kept explicit."
          />
          <div className="table-heading">
            <div>
              <h3>Explore the benchmark</h3>
              <p>Choose a condition. Sort a metric. Inspect a model.</p>
            </div>
            <span className="pill">Eight VLMs / Shared protocol</span>
          </div>
          <Results />
          <BehaviorProfiles />
          <Findings />
          <Reasoning />
          <Training />
        </section>
        <section id="demo" className="section demo-section">
          <SectionHead
            n="05"
            label="THE PROJECT FILM"
            title={
              <>
                Safety lives
                <br />
                between observation and action.
              </>
            }
            description="A conference-ready walkthrough of the motivation, benchmark, recorded agent behavior, and main findings."
          />
          <div className="film-toolbar">
            <div className="segmented" aria-label="Film version">
              <button
                aria-pressed={film === "nyc-90s"}
                onClick={() => setFilm("nyc-90s")}
              >
                NYC project film · 1:30
              </button>
              <button
                aria-pressed={film === "nyc-comparison"}
                onClick={() => setFilm("nyc-comparison")}
              >
                Astra vs. Sol · 0:44
              </button>
            </div>
            <span className="micro">RT–SAFE / PROJECT PRESENTATION</span>
          </div>
          <div className="film-shell">
            <video
              key={film}
              ref={videoRef}
              controls
              playsInline
              preload="none"
              poster={film === "nyc-90s" ? "media/nyc/film-poster.webp" : "media/nyc/comparison-poster.webp"}
              aria-label="RT-SAFE project film"
            >
              <source src={`media/rt-safe-${film}.mp4`} type="video/mp4" />
              <track
                kind="captions"
                src={`media/rt-safe-${film}.vtt`}
                srcLang="en"
                label="English"
                default
              />
              Your browser does not support video. Download the MP4 below.
            </video>
          </div>
          <div className="film-footer">
            <span>{film === "nyc-90s" ? "1920 × 1080 · Narrated · English captions" : "1920 × 1080 · 6× playback · English captions"}</span>
            <div>
              <a href={`media/rt-safe-${film}.mp4`} download>
                Download film <span>↓</span>
              </a>
              <a href={`media/rt-safe-${film}.vtt`} download>
                Captions <span>↓</span>
              </a>
              <a href="media/rt-safe-nyc-90s-captioned.mp4" download>
                90s conference edition <span>↓</span>
              </a>
            </div>
          </div>
        </section>
        <section id="resources" className="section resources-section">
          <SectionHead
            n="06"
            label="GO DEEPER"
            title={<>Built to be inspected.</>}
            description="Read the protocol, download the results, and trace the examples back to their recorded decisions."
          />
          <div className="resource-grid">
            <a href="media/rt-safe-paper.pdf" target="_blank" rel="noreferrer">
              <span className="resource-icon">01</span>
              <h3>
                The manuscript <Arrow diagonal />
              </h3>
              <p>Benchmark design, experiments, and implementation details.</p>
              <span className="micro">PDF / RESEARCH MANUSCRIPT</span>
            </a>
            <a href="data/results.csv" download>
              <span className="resource-icon">02</span>
              <h3>
                The results <span>↓</span>
              </h3>
              <p>
                Eight models across real-time, static, and averaged conditions.
              </p>
              <span className="micro">CSV / ALSO AVAILABLE AS JSON</span>
            </a>
            <a href="data/methodology.txt" download>
              <span className="resource-icon">03</span>
              <h3>
                The evaluation protocol <span>↓</span>
              </h3>
              <p>
                Metrics, timing, aggregation, and the limits of each comparison.
              </p>
              <span className="micro">PLAIN TEXT / METHODOLOGY</span>
            </a>
          </div>
          {repositoryUrl && (
            <a className="button" href={repositoryUrl} target="_blank" rel="noreferrer">
              Benchmark source code <Arrow diagonal />
            </a>
          )}
          <details className="methodology">
            <summary>
              <span>How to interpret these results</span>
              <b>+</b>
            </summary>
            <div className="methodology-content">
              <div>
                <h4>Simulator-derived events</h4>
                <p>
                  Safety is evaluated from simulator counters, overlap triggers,
                  and traffic checks. Sustained physical contact may produce
                  repeated counts. Collisions are event counts, not counts of
                  distinct injured people.
                </p>
              </div>
              <div>
                <h4>Timing and prompt context</h4>
                <p>
                  Static mode pauses inference; real-time mode does not. The
                  prompts also describe the timing mode, so this comparison is a
                  joint change in timing and instructions. Response time depends
                  on the serving interface.
                </p>
              </div>
              <div>
                <h4>Implementation scope</h4>
                <p>
                  Hazards are detected at post-action endpoints. Water is a
                  recorded flag with no movement perturbation in the reported
                  implementation. Results describe these simulated tasks,
                  without claiming real-world safety certification.
                </p>
              </div>
              <div>
                <h4>Presentation sources</h4>
                <p>
                  Tables come from the supplied manuscript. Decision images are
                  actual UE captures. The opening city illustration and social
                  card are AI-generated; timing schematics are explanatory.
                </p>
              </div>
            </div>
          </details>
        </section>
        <section className="closing">
          <div className="eyebrow">
            <i /> RT–SAFE
          </div>
          <h2>
            Evaluate the whole decision.
            <br />
            <span>Including the time it takes.</span>
          </h2>
          <a className="button" href="#results">
            Explore the evidence <Arrow />
          </a>
        </section>
        <footer className="footer">
          <a className="brand" href="#top">
            <span className="brand-icon">rt</span> RT–SAFE
          </a>
          <p>Benchmarking agent safety in real-time embodied environments.</p>
          <a href="#top">Back to top ↑</a>
        </footer>
      </main>
    </>
  );
}
