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
export function BehaviorProfiles() {
  return <section className="behavior-profiles" aria-labelledby="behavior-heading">
    <div className="table-heading"><div><h3 id="behavior-heading">What lies behind the leaderboard?</h3><p>Navigation choices shape how long an agent is exposed to a moving world.</p></div><span className="pill">Paper Figures 3 & 7</span></div>
    <div className="behavior-grid">{cards.map(card => <article className="behavior-card" key={card.name}>
      <header><span className="behavior-model" style={{color:card.color}}><ModelLogo name={card.name} />{card.name}</span><h4>{card.title}</h4></header>
      <img src={`media/behavior/${card.name.toLowerCase()}.svg`} alt={`${card.name}: ${data.axes.join(", ")}; values normalized across all eight models.`} width="900" height="900" loading="lazy" />
      <p>{card.text}</p>
    </article>)}</div>
    <p className="behavior-note">All difficulties, provider-default reasoning, 108 episodes per model. Each axis is normalized across all eight models. Move length means commanded distance. Waiting excludes inference time. Radar area is not an overall safety score. <a href="data/behavior.json" download>Download profile data ↗</a></p>
  </section>;
}
