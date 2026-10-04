import { useEffect, useRef, useState } from "react";

const media = "media/nyc/";

export function NYCHero() {
  const ref = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const video = ref.current;
    video?.play().then(() => setPlaying(true)).catch(() => {});
    return () => video?.pause();
  }, []);
  return <>
    <video ref={ref} className="hero-image nyc-hero-video" muted loop playsInline
      preload="metadata" poster={`${media}hero.webp`} aria-label="Unreal Engine view of the Madison Square Park NYC scene"
      onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}>
      <source src={`${media}hero.mp4`} type="video/mp4" />
    </video>
    <button className="nyc-motion-toggle" aria-label={playing ? "Pause city scene" : "Play city scene"}
      onClick={() => playing ? ref.current?.pause() : void ref.current?.play().catch(() => {})}>
      {playing ? "Ⅱ Pause scene" : "▶ Play scene"}
    </button>
  </>;
}

function SynchronizedViews({ comparison = false }: { comparison?: boolean }) {
  const left = useRef<HTMLVideoElement>(null);
  const right = useRef<HTMLVideoElement>(null);
  const requestedTime = useRef(0);
  const playGeneration = useRef(0);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const duration = comparison ? 44.1 : 11;
  const clipMedia = comparison ? "media/recorded/" : media;
  const stems = comparison ? ["astra", "sol"] : ["timing-static", "timing-realtime"];
  const labels = comparison ? ["GPT-6 Astra", "GPT-5.6 Sol"] : ["Static evaluation", "Real-time evaluation"];
  const pause = () => { playGeneration.current++; left.current?.pause(); right.current?.pause(); setPlaying(false); setLoading(false); };
  const seek = (value: number) => {
    pause();
    requestedTime.current = value;
    for (const ref of [left, right]) if (ref.current?.readyState) ref.current.currentTime = Math.min(value, ref.current.duration - .001);
    setTime(value);
  };
  const play = async () => {
    setError("");
    if (time >= duration - .15) seek(0);
    const generation=++playGeneration.current;
    setLoading(true);
    try {
      await Promise.all([left.current, right.current].map(video => new Promise<void>((resolve,reject) => {
        if (!video) { reject(new Error("Video missing")); return; }
        if (video.readyState>=3) { resolve(); return; }
        const cleanup=()=>{clearTimeout(timer);video.removeEventListener("canplay",ready);video.removeEventListener("error",failed);};
        const ready=()=>{cleanup();resolve();};
        const failed=()=>{cleanup();reject(new Error("Video could not load"));};
        const timer=setTimeout(failed,15000);
        video.addEventListener("canplay",ready,{once:true});video.addEventListener("error",failed,{once:true});video.load();
      })));
      if (generation!==playGeneration.current) return;
      await Promise.all([left.current?.play(), right.current?.play()]);
      if (generation!==playGeneration.current) { left.current?.pause();right.current?.pause();return; }
      setPlaying(true);
    } catch {
      if (generation===playGeneration.current) { pause(); setError("The videos could not play. Reload the page or use the film download."); }
    } finally {
      if (generation===playGeneration.current) setLoading(false);
    }
  };
  useEffect(() => {
    if (!playing) return;
    let frame: number;
    const tick = () => {
      const a=left.current, b=right.current;
      if (a && b) {
        setTime(b.currentTime);
        if (Math.abs(a.currentTime-b.currentTime)>.15 && !a.ended) a.currentTime=b.currentTime;
        if (b.ended) { a.pause(); setPlaying(false); return; }
      }
      frame=requestAnimationFrame(tick);
    };
    frame=requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);
  useEffect(() => () => { playGeneration.current++;left.current?.pause(); right.current?.pause(); }, []);
  return <div className={`nyc-pair ${comparison ? "nyc-case" : "nyc-timing"}`}>
    <div className="nyc-pair-grid">
      {stems.map((stem,i) => <article key={stem} className={`nyc-view nyc-view-${i}`}>
        <div className="nyc-view-frame"><video ref={i === 0 ? left : right} muted playsInline preload="none"
          poster={`${clipMedia}${stem}.webp`} aria-label={`${labels[i]} ${comparison ? "original first-person recording" : "in the NYC scene"}`}
          onLoadedMetadata={event => { const v=event.currentTarget; v.currentTime=Math.min(requestedTime.current,Math.max(0,v.duration-.001)); }}>
          <source src={`${clipMedia}${stem}.mp4?v=silent-motion-2`} type="video/mp4" />
        </video></div>
      </article>)}
    </div>
    <div className="nyc-transport">
      <button className="nyc-play" disabled={loading} onClick={() => playing ? pause() : void play()} aria-label={loading ? "Loading both views" : playing ? "Pause both views" : "Play both views"}>{loading ? "Loading…" : playing ? "Ⅱ Pause" : time >= duration-.15 ? "↻ Replay" : "▶ Play both"}</button>
      <label className="nyc-scrubber"><span className="sr-only">{comparison ? "Comparison playback time" : "Timing illustration playback time"}</span>
        <input type="range" min="0" max={duration} step=".05" value={time} onChange={e=>seek(Number(e.target.value))} />
      </label>
      <span className="nyc-time">{time.toFixed(1)} / {duration.toFixed(1)} s</span>
    </div>
    {error && <p role="alert" className="fineprint">{error}</p>}
    <p className="fineprint">{comparison
      ? "Final-subgoal segment from the same RT15 task, easy difficulty, seed 0, low reasoning. Original first-person observations and action snapshots play on the recorded simulation timeline at 6× speed. Brief dissolves soften snapshot changes; inputs then hold during inference. This is a snapshot replay. Collision alerts appear at report times in the source logs."
      : "Native Unreal rendering of the same initial scene and commanded move. Timing and the encounter are illustrative; benchmark results are measured separately."}</p>
    {comparison && <a className="nyc-evidence-link" href="data/recorded-replay.json" download>Inspect the recorded actions and timing ↗</a>}
  </div>;
}

export function NYCTiming() {
  return <figure className="nyc-environment nyc-timing-story">
    <video controls playsInline preload="none" poster={`${media}timing-story.webp?v=silent-motion-2`} aria-label="From static evaluation to real-time evaluation in Madison Square Park">
      <source src={`${media}timing-story.mp4?v=silent-motion-2`} type="video/mp4" />
      <track kind="captions" src={`${media}timing-story.vtt?v=silent-motion-2`} srcLang="en" label="English" default />
    </video>
    <figcaption><strong>The real world does not stop.</strong><span>Static evaluation → RT-Safe · illustrative NYC encounter</span></figcaption>
  </figure>;
}
export function NYCComparison() { return <SynchronizedViews comparison />; }

export function NYCEnvironment() {
  return <figure className="nyc-environment">
    <video controls playsInline preload="none" poster={`${media}environment-intro.webp?v=silent-motion-2`} aria-label="NYC environment tour showing the route and safety events">
      <source src={`${media}environment-intro.mp4?v=silent-motion-2`} type="video/mp4" />
      <track kind="captions" src={`${media}environment-intro.vtt?v=silent-motion-2`} srcLang="en" label="English" default />
    </video>
    <figcaption><strong>Our RT-Safe task: reach the goal safely.</strong><span>NYC demonstration scene · pedestrians, obstacles, hazards, and traffic rules</span></figcaption>
  </figure>;
}

const views=[
  {id:"skyline",label:"The city",description:"An aerial view of the Madison Square Park scene."},
  {id:"park",label:"Park & blocks",description:"The park and surrounding streets from above."},
  {id:"high_follow",label:"Agent view",description:"An elevated third-person view of the navigation corridor."},
  {id:"intersection",label:"The crossing",description:"The intersection and its marked crossing."},
];
export function NYCSceneGallery() {
  const [active,setActive]=useState(0);
  return <details className="nyc-gallery"><summary>Explore the NYC scene <span>Four camera views ↗</span></summary>
    <div className="nyc-gallery-inner">
      <div className="nyc-camera-tabs" aria-label="Scene camera">{views.map((v,i)=><button key={v.id} aria-pressed={active===i} onClick={()=>setActive(i)}>{v.label}</button>)}</div>
      <img src={`${media}camera-${views[active].id}.webp`} alt={views[active].description} width="1920" height="1080" loading="lazy" />
      <p>Madison Square Park, NYC · rendered in Unreal Engine with Movie Render Queue.</p>
    </div>
  </details>;
}
