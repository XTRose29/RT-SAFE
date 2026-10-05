import { useEffect, useRef, useState } from "react";
import evidence from "./model-examples.json";
import { ModelLogo } from "./model-logos";
import { BehaviorProfile } from "./behavior-profiles";
import { NYCComparison } from "./nyc-scenes";

type Example = typeof evidence.models[number];
const order = ["Astra", "Sol", "Sonnet", "Fable", "Gemini", "DeepSeek", "Inkling", "Grok"];
export const modelExamples = order.map(name => evidence.models.find(m => m.name === name)!);

export function RecordedExample({ model }: { model: Example }) {
  const video = useRef<HTMLVideoElement>(null);
  const pendingSeek = useRef<number | null>(null);
  useEffect(() => { const player = video.current; return () => player?.pause(); }, []);
  const [time, setTime] = useState(0);
  const [error, setError] = useState("");
  const jump = (seconds: number) => {
    const v = video.current;
    if (!v) return;
    setError("");
    if (v.readyState > 0) v.currentTime = seconds;
    else { pendingSeek.current = seconds; v.load(); }
    void v.play().catch(() => setError("Press play to continue the example."));
  };
  const active = model.steps.findIndex(s => time * model.speed < s.end);
  return <div className="recorded-example recorded-example-compact">
    <div className="example-player">
      <video ref={video} controls muted playsInline preload="none" poster={model.poster}
        aria-label={`${model.name} recorded behavior example`}
        onPlay={e => document.querySelectorAll<HTMLVideoElement>(".recorded-example video").forEach(v => { if (v !== e.currentTarget) v.pause(); })}
        onTimeUpdate={e => setTime(e.currentTarget.currentTime)}
        onLoadedMetadata={e => { if (pendingSeek.current !== null) { e.currentTarget.currentTime = pendingSeek.current; pendingSeek.current = null; } }}
        onError={() => setError("This video could not load. Try the direct video link below.")}>
        <source src={model.video} type="video/mp4" />
      </video>
    </div>
    <div className="example-strip">
      <span className="example-condition">Recorded excerpt · {model.difficulty} · Task {model.task} · {model.speed}×</span>
      <strong>{model.headline}</strong>
      <span>{model.contacts} contact{model.contacts === 1 ? "" : "s"} · {model.hazards > 0 && `${model.hazards} hazard interaction${model.hazards === 1 ? "" : "s"} · `}{model.trafficViolations > 0 && `${model.trafficViolations} traffic violations · `}{model.sourceDuration.toFixed(1)} simulation seconds · 3 decisions</span>
    </div>
    <details className="example-record-details"><summary>Explore recorded decisions & events</summary>
    <div className="example-detail">
      <div className="example-condition">{model.difficulty} · Task {model.task} · {model.reasoning}</div>
      <h4>{model.headline}</h4>
      <p>{model.rationale}</p>
      <dl className="example-stats">
        <div><dt>Response / decision</dt><dd>{model.meanResponse.toFixed(1)} s</dd></div>
        <div><dt>Contacts in excerpt</dt><dd>{model.contacts}<small>{model.inferenceContacts} during inference · {model.actionContacts} during action</small></dd></div>
      </dl>
      {model.hazards + model.trafficViolations > 0 && <p className="example-other-events">Also recorded: {model.hazards} hazard interaction{model.hazards === 1 ? "" : "s"} · {model.trafficViolations} traffic violations.</p>}
      <div className="example-chapters" aria-label={`${model.name} recorded decisions`}>
        <span className="micro">EXPLORE THE THREE DECISIONS</span>
        {model.steps.map((step, i) => <button key={step.decision} aria-pressed={i === active} onClick={() => jump(step.start / model.speed)}>
          <span>▶ Decision {step.decision}</span><strong>{step.command}</strong><small>{step.responseSeconds.toFixed(1)} s model response</small>
        </button>)}
      </div>
      {model.events.length > 0 && <button className="example-contact-jump" onClick={() => jump(Math.max(0, model.events[0].at - .3))}>↗ Jump to first collision report</button>}
      <p className="example-provenance">Original benchmark map · {model.map.replace("map", "Map ").replace("_", " / ")} · seed {model.seed}. Decisions {model.sourceDecisions[0]}–{model.sourceDecisions.at(-1)}. {model.sourceDuration.toFixed(1)} simulation seconds at {model.speed}× speed.</p>
      <p className="example-provenance">Snapshot replay with brief dissolves. The current input holds during inference; collision alerts follow phase reports. Counts cover this excerpt only.</p>
      <a className="example-download" href={model.video}>Open {model.name} video ↗</a>
      {error && <p role="alert">{error}</p>}
    </div>
    </details>
  </div>;
}

export function ModelExamples() {
  const [selected, setSelected] = useState("Astra");
  const [comparison, setComparison] = useState(false);
  const model = modelExamples.find(m => m.name === selected)!;
  return <div className="model-examples">
    <div className="example-model-selector" role="group" aria-label="Choose a model example">
      {modelExamples.map(m => <button key={m.name} aria-pressed={selected === m.name && !comparison} onClick={() => { setSelected(m.name); setComparison(false); }}><ModelLogo name={m.name} /><span>{m.name}</span></button>)}
    </div>
    <div className="example-heading"><span className="micro">{comparison ? "MATCHED ROUTE COMPARISON · LOW REASONING" : "MODEL BEHAVIOR · PROVIDER-DEFAULT REASONING"}</span><button className="example-compare" aria-pressed={comparison} onClick={() => setComparison(!comparison)}>{comparison ? "← Back to model examples" : "Compare Astra & Sol on the same route ↗"}</button></div>
    {comparison ? <NYCComparison /> : <div className="model-overview" key={selected}>
      <RecordedExample model={model} />
      <BehaviorProfile name={selected} />
    </div>}
    <p className="example-selection-note">Each excerpt illustrates a behavior seen in the model’s profile. Tasks and difficulties differ, so these clips are qualitative examples. Use the leaderboard for aggregate comparisons. <a href="data/model-examples.json" download>Inspect selection, actions & source hashes ↗</a></p>
  </div>;
}
