import { useState } from "react";
import { ModelLogo } from "./model-logos";
import data from "./behavior.json";
const cards = [
  { name: "Inkling", color: "#518b37", title: "Longer moves, fewer decisions.", text: "Despite a 26.0 s response latency, Inkling uses the fewest decisions per episode: 38.5. Its collisions are comparable to much faster models." },
  { name: "Grok", color: "#c23b52", title: "Slow responses add exposure.", text: "Grok combines long response times with shorter commanded moves than Inkling. It records the most collisions: 59.5 per episode." },
  { name: "Astra", color: "#16878a", title: "Less turning and waiting.", text: "Astra turns and waits less than Fable and uses fewer decisions: 51.7 versus 58.9 per episode." },
  { name: "Fable", color: "#9557b4", title: "More turns and waits, fewer contacts.", text: "Fable turns and waits more than Astra, with slightly fewer collisions: 20.9 versus 22.4 per episode." },
  { name: "Sonnet", color: "#cb6b9c", title: "Fast responses, fewer collisions.", text: "Sonnet responds in 4.9 s on average, the fastest of the eight models. It also records the fewest collisions: 19.6 per episode." },
  { name: "Sol", color: "#357abb", title: "More decisions, turns, and waits.", text: "Sol makes the most decisions per episode: 69.1. Its action mix has the most turning and waiting, with shorter commanded moves." },
  { name: "Gemini", color: "#b88b18", title: "Shorter moves and frequent turns.", text: "Gemini favors relatively short commanded moves and frequent turns. Its mean response latency is 11.2 s, with 27.7 collisions per episode." },
  { name: "DeepSeek", color: "#ba6f3e", title: "Slow responses, high collision count.", text: "DeepSeek averages 40.4 s per response and 51.4 collisions per episode. Its commanded moves are longer than Grok’s, but shorter than Inkling’s." },
];
const axisMetrics = [
  { label: "Collisions / episode", unit: "contacts", note: "Mean recorded contacts per episode, including unsuccessful episodes. Fewer contacts extend the radar outward." },
  { label: "Response latency", unit: "s / decision", note: "Time spent waiting for a model response. Faster responses extend the radar outward." },
  { label: "Decisions / episode", unit: "decisions", note: "Mean decisions made during an episode. Fewer decisions extend the radar outward." },
  { label: "Commanded move length", unit: "m / move", note: "Requested movement distance, averaged over movement actions. This is not realized route progress." },
  { label: "Wait actions", unit: "% of actions", note: "Explicit wait actions as a share of action choices. Time spent on inference is excluded." },
  { label: "Turn actions", unit: "% of actions", note: "Turn actions as a share of all action choices." },
];
export function BehaviorProfile({ name }: { name: string }) {
  const card = cards.find(card => card.name === name)!;
  const [axis, setAxis] = useState(0);
  const profile = data.profiles.find(p => p.name === card.name)!;
  const raw = [profile.average.collisions, profile.average.latency, profile.average.decisions, profile.raw[3], profile.raw[4] * 100, profile.raw[5] * 100];
  const point = (i: number, r: number) => [210 + Math.sin(i * Math.PI / 3) * r, 180 - Math.cos(i * Math.PI / 3) * r];
  const polygon = (values: number[]) => values.map((v, i) => point(i, v * 112).join(",")).join(" ");
  const metric = axisMetrics[axis];
  return <article className="behavior-card">
    <header><span className="behavior-model" style={{color:card.color}}><ModelLogo name={card.name} />{card.name}</span><h4>{card.title}</h4></header>
    <p className="profile-scope">Aggregate behavior · 108 episodes<br />All difficulties · provider-default reasoning</p>
    <svg className="interactive-radar" viewBox="0 0 420 350" role="group" aria-label={`${card.name} interactive behavior profile`}>
      {[.25,.5,.75,1].map(r => <polygon key={r} points={polygon(Array(6).fill(r))} fill="none" stroke="#d7dfdf" />)}
      {data.axes.map((label, i) => { const [x,y] = point(i,112); return <line key={label} x1="210" y1="180" x2={x} y2={y} stroke="#d7dfdf" />; })}
      <polygon points={polygon(profile.values)} fill={card.color} fillOpacity=".16" stroke={card.color} strokeWidth="2.5" />
      {data.axes.map((label, i) => {
        const [x,y] = point(i,profile.values[i]*112);
        const [tx,ty] = point(i,143);
        const words = label.split(" "); const half = Math.ceil(words.length/2);
        return <g key={label} className="radar-axis" role="button" tabIndex={0} aria-label={`${label}: ${raw[i].toFixed(i > 2 ? 2 : 1)} ${axisMetrics[i].unit}`} aria-pressed={axis===i}
          onMouseEnter={() => setAxis(i)} onFocus={() => setAxis(i)} onClick={() => setAxis(i)} onKeyDown={e => { if(e.key === "Enter" || e.key === " "){e.preventDefault();setAxis(i);} }}>
          <line x1={x} y1={y} x2={tx} y2={ty} stroke="transparent" strokeWidth="26" />
          <circle cx={x} cy={y} r="18" fill="transparent" />
          <circle cx={x} cy={y} r={axis===i ? 6 : 3} fill={card.color} stroke="white" strokeWidth="2" />
          <text x={tx} y={ty-5} textAnchor="middle" fill={axis===i ? card.color : "#41534e"} fontSize="12" fontWeight={axis===i ? "700" : "400"}>
            <tspan x={tx}>{words.slice(0,half).join(" ")}</tspan><tspan x={tx} dy="16">{words.slice(half).join(" ")}</tspan>
          </text>
        </g>;
      })}
    </svg>
    <div className="radar-readout" aria-live="polite"><span>{metric.label}</span><strong style={{color:card.color}}>{raw[axis].toFixed(axis > 2 ? 2 : 1)} <small>{metric.unit}</small></strong><p>{metric.note}</p></div>
    <p>{card.text}</p>
    <details className="profile-details"><summary>Explore {card.name} results</summary>
      <dl>{[
        ...axisMetrics.map((m,i) => [m.label, `${raw[i].toFixed(i>2 ? 2 : 1)} ${m.unit}`]),
        ["Success rate", `${profile.average.success.toFixed(1)}%`],
        ["Safe success rate", `${profile.average.safeSuccess.toFixed(1)}%`],
        ["SPL", profile.average.spl.toFixed(3)],
        ["Contacts during inference", `${profile.average.passiveShare.toFixed(1)}%`],
      ].map(([label,value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      <p>Real-time · all difficulties · provider-default reasoning · 108 episodes. Source: manuscript Figures 3 & 7 and Appendix B.3.</p>
    </details>
    <p className="behavior-note">Tap a radar axis to inspect its value. Axes are normalized across all eight models; radar area is not an overall safety score. <a href="data/behavior.json" download>Profile data ↗</a></p>
  </article>;
}
