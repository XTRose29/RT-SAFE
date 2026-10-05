const labels = [
  { x: 76, y: 196, w: 370, color: "#ffbaad", title: "03  Traffic-rule violations", lines: ["Use the marked crossing.", "Wait for the pedestrian signal."], targets: [[404,178],[699,399]] },
  { x: 1445, y: 356, w: 390, color: "#a9dff4", title: "01  Collisions", lines: ["People, robot dogs, vehicles", "and obstacles share the route."], targets: [[1075,593],[1090,490],[1186,660]] },
  { x: 1445, y: 675, w: 390, color: "#f9cf85", title: "02  Hazard interactions", lines: ["Avoid oil, water and trip hazards."], targets: [[1143,704],[1014,602],[1129,542]] },
];
export function EnvironmentOverview({ onStart, buttonRef }: { onStart: () => void; buttonRef: React.RefObject<HTMLButtonElement | null> }) {
  return <figure className="environment-overview">
    <div className="overview-scene">
      <img src="media/nyc/environment-overview.webp" width="1920" height="1080" loading="lazy" alt="Native NYC scene: an agent starts on the sidewalk beside oil, water, a trip hazard, pedestrians, a robot dog and boxes. The route leads across the marked crossing to the goal, with traffic and signals at the intersection." />
      <svg className="overview-annotations" viewBox="0 0 1920 1080" aria-hidden="true">
        <defs><marker id="overview-arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="none" stroke="#bce88c" strokeWidth="1.7" /></marker></defs>
        <path d="M1040 895 L984 715 L954 598 L974 470 L957 403 L628 390 L252 382" fill="none" stroke="#bce88c" strokeWidth="6" strokeDasharray="16 12" markerEnd="url(#overview-arrow)" />
        <g className="overview-title"><rect x="56" y="42" width="840" height="105" rx="12" fill="#142e29" fillOpacity=".96" /><text x="83" y="88" fill="white" fontSize="35" fontWeight="600">Our RT-SAFE task &amp; environment</text><text x="83" y="126" fill="#d6e3d5" fontSize="24">Reach the goal with zero recorded safety events.</text></g>
        {labels.map(l => <g key={l.title}>
          {l.targets.map(([x,y],i) => <g key={i}><path d={`M${l.x > 1000 ? l.x : l.x + l.w} ${l.y + 48 + i * 17} L${x} ${y}`} fill="none" stroke={l.color} strokeWidth="2.5" /><circle cx={x} cy={y} r="12" fill="#162d28" stroke={l.color} strokeWidth="3" /></g>)}
          <rect x={l.x} y={l.y} width={l.w} height={l.lines.length===1?108:140} rx="12" fill="#142e29" fillOpacity=".96" stroke={l.color} strokeWidth="2" />
          <text x={l.x+22} y={l.y+42} fill={l.color} fontWeight="600" fontSize="27">{l.title}</text>
          {l.lines.map((line,i) => <text key={line} x={l.x+22} y={l.y+81+i*29} fill="#eef4ec" fontSize="23">{line}</text>)}
        </g>)}
        <g fill="#142e29" stroke="#bce88c" strokeWidth="2"><rect x="966" y="927" width="148" height="53" rx="9" /><rect x="159" y="396" width="146" height="53" rx="9" /></g>
        <g fill="#d6f8ad" fontSize="27" fontWeight="600" textAnchor="middle"><text x="1040" y="963">Start</text><text x="232" y="432">Goal</text></g>
      </svg>
      <button className="overview-start" ref={buttonRef} onClick={onStart}><span className="overview-start-icon" aria-hidden="true">↔</span><span>Start exploring RT-SAFE<small>Drag to look around · Choose a viewpoint</small></span><span aria-hidden="true">↗</span></button>
    </div>
    <figcaption className="overview-caption"><div><strong>One goal. Three dimensions of safety.</strong><span>Explore the NYC scene and inspect what can go wrong along the way.</span></div><span className="overview-provenance">Native Unreal Engine render · illustrative route</span></figcaption>
    <div className="overview-mobile-key"><span><i />Collisions</span><span><i />Hazards</span><span><i />Traffic rules</span></div>
  </figure>;
}
