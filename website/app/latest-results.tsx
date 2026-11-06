import { useMemo, useState } from "react";
import snapshot from "./latest-results.json";
import "./latest-results.css";

type Metric = "latency" | "decisions" | "safetyEvents";
const metrics: { key: Metric; title: string; unit: string }[] = [
  { key: "latency", title: "Response time", unit: "s / response" },
  { key: "decisions", title: "Decisions", unit: "per task" },
  { key: "safetyEvents", title: "Safety events", unit: "per task" },
];
const fixed = (value: number) => value.toFixed(1);
const model = (slug: string) => snapshot.models.find(row => row.slug === slug)!;
export function LatestResults() {
  const [family, setFamily] = useState("gemini");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Metric | "family">("family");
  const pair = snapshot.pairs.find(row => row.key === family)!;
  const old = model(pair.older), newer = model(pair.newer);
  const rows = useMemo(() => snapshot.models.filter(row => `${row.name} ${row.provider}`.toLowerCase().includes(query.toLowerCase())).sort((a,b) => sort === "family" ? 0 : a[sort] - b[sort]), [query, sort]);
  return <section id="results" className="section latest-results" aria-labelledby="latest-title">
    <div className="section-head">
      <div className="eyebrow"><span className="section-number">RQ1</span> LATEST MODEL RESULTS</div>
      <div className="section-title"><h2 id="latest-title">Newer agents.<br />Safer in real time?</h2><p>Compare model versions on shared routes. See where response speed, decision count, and safety outcomes improve—and where they don’t.</p></div>
    </div>
    <div className="latest-scope"><span><strong>{snapshot.models.length}</strong> model versions</span><span><strong>10</strong> planned routes / model</span><span><strong>{snapshot.validatedEpisodes}/{snapshot.plannedEpisodes}</strong> validated episodes</span><span>Hard · Real-time · Seed 0</span></div>
    <nav className="latest-links" aria-label="Results sections"><a href="#version-comparison">Old → new</a><a href="#latest-model-table">All model results</a><a href="#evaluation-comparisons">Static vs. real time & reasoning effort ↘</a></nav>
    <div id="version-comparison" className="version-comparison">
      <div className="table-heading"><div><h3>What changed between versions?</h3><p>Each comparison uses only routes validated for both models. Lower values mean less time, fewer decisions, or fewer recorded events.</p></div></div>
      <div className="version-selector" role="group" aria-label="Model version comparison">{snapshot.pairs.map(row => <button key={row.key} aria-pressed={family === row.key} onClick={() => setFamily(row.key)}>{row.label}<span>{row.tasks.length} matched routes</span></button>)}</div>
      <div className="version-panel" aria-live="polite" aria-atomic="true">
        <div className="version-names"><div><span className="micro">REFERENCE</span><h4>{old.name}</h4><p>{old.effort} reasoning · {old.provider}</p></div><span aria-hidden="true">→</span><div><span className="micro">COMPARISON</span><h4>{newer.name}</h4><p>{newer.effort} reasoning · {newer.provider}</p></div></div>
        <div className="version-metrics">{metrics.map(({key,title,unit}) => {
          const delta = pair.changes[key], max = Math.max(pair.old[key],pair.new[key]);
          const near = Math.abs(delta) < 1;
          return <article className="version-metric" key={key}>
            <h5>{title}</h5><strong className={`version-delta ${near ? "neutral" : delta < 0 ? "improved" : "increased"}`}>{near ? "~ Unchanged" : `${Math.round(Math.abs(delta))}% ${delta < 0 ? "lower" : "higher"}`}</strong>
            <p className="version-values">{fixed(pair.old[key])} <span>→</span> {fixed(pair.new[key])} <small>{unit}</small></p>
            <div className="version-bars" aria-hidden="true"><div className="reference-bar" style={{width:`${pair.old[key]/max*100}%`}} /><div className="comparison-bar" style={{width:`${pair.new[key]/max*100}%`}} /></div>
          </article>;
        })}</div>
        <div className="version-outcomes"><span>Goals reached <strong>{pair.old.goals}/{pair.old.count} → {pair.new.goals}/{pair.new.count}</strong></span><span>Safe successes <strong>{pair.old.safe}/{pair.old.count} → {pair.new.safe}/{pair.new.count}</strong></span><span className="version-legend"><i /> Reference <i /> Comparison</span></div>
        <p className="version-note">{pair.note}</p>
      </div>
    </div>
    <div className="latest-takeaway"><strong>Safety failures persist.</strong><p>Reaching the goal is not the same as reaching it safely. Efficiency gains do not consistently translate into fewer safety events.</p></div>
    <div id="latest-model-table" className="table-heading"><div><h3>All model results</h3><p>Per-model validated coverage is shown below. Partial rows cover different routes and are not a complete leaderboard.</p></div><a className="text-link" href="data/latest-results.csv" download>CSV ↓</a></div>
    <div className="latest-table-tools"><label>Find a model<input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Model or provider…" /></label><label>Order by<select value={sort} onChange={e=>setSort(e.target.value as typeof sort)}><option value="family">Model family</option>{metrics.map(m=><option key={m.key} value={m.key}>{m.title} · lower first</option>)}</select></label></div>
    <div className="table-scroll latest-table" role="region" aria-label="Latest model results, horizontally scrollable" tabIndex={0}><table>
      <caption className="sr-only">Latest hard real-time results. Coverage varies for older DeepSeek and GPT-6 Astra. Latency is pooled per response; decisions and safety events are per-task averages.</caption>
      <thead><tr><th scope="col">Model / reasoning</th><th scope="col">Validated</th><th scope="col">Goals</th><th scope="col">Safe successes</th><th scope="col">Response time</th><th scope="col">Decisions / task</th><th scope="col">Safety events / task</th></tr></thead>
      <tbody>{rows.map(row=><tr key={row.slug}><th scope="row"><strong>{row.name}</strong><small>{row.effort} · {row.provider}</small></th><td>{row.count}/{row.expected}{row.count < row.expected && <small className="partial-coverage">Partial</small>}</td><td>{row.goals}/{row.count}</td><td>{row.safe}/{row.count}</td><td>{fixed(row.latency)} s</td><td>{fixed(row.decisions)}</td><td>{fixed(row.safetyEvents)}</td></tr>)}</tbody>
    </table>{rows.length === 0 && <p className="latest-empty">No models match your search.</p>}</div>
    <p className="latest-definition"><strong>What counts as a safety event?</strong> Contacts during movement or inference, trip/oil/water interactions, and traffic violations. Repeated contacts may count more than once. Safe success means reaching the goal with zero recorded safety events.</p>
    <details className="latest-methods"><summary>Coverage, settings & data provenance</summary><div><p>{snapshot.aggregation}</p><p>{snapshot.limitations}</p><p><strong>Older DeepSeek:</strong> {snapshot.missing.deepseek_old}</p><p><strong>GPT-6 Astra:</strong> {snapshot.missing.gpt_astra}</p><p>Model-specific reasoning and serving providers are listed above. The earlier reasoning-effort and static/real-time comparisons use separate evaluation sets and remain available below.</p><p>Three historical Kimi K3.1 episodes use preserved audited metrics because their original result files are unavailable; these records are identified in the download.</p><p>Snapshot prepared {snapshot.generatedAt.slice(0,10)} · <a href="data/latest-results.json" download>Download metrics, episode records & source hashes (JSON) ↓</a></p></div></details>
  </section>;
}
