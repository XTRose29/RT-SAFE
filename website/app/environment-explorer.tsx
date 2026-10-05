import { useEffect, useRef, useState } from "react";
import { EnvironmentOverview } from "./environment-overview";
import { Panorama, type ViewAngle } from "./panorama";

type Location = { id: string; label: string; position: number[]; yaw: number };
type Scene = { locations: Location[]; anchors: Record<string, number[]> };
const categories = [
  {id:"collisions",label:"Collisions",anchor:"pedestrians",view:"approach",title:"Give moving actors room.",description:"Contact with people, robot dogs, vehicles, and objects counts as a collision."},
  {id:"hazards",label:"Hazards",anchor:"oil",view:"hazards",title:"Look beyond the shortest path.",description:"Oil, water, and trip hazards make a direct route unsafe. Look down to inspect the path."},
  {id:"rules",label:"Traffic rules",anchor:"crossing",view:"crossing",title:"Use the crossing. Follow the signal.",description:"Entering the road outside a marked crossing or crossing on red is a traffic-rule violation."},
];
const clamp=(v:number,a:number,b:number)=>Math.max(a,Math.min(b,v));
export function EnvironmentExplorer() {
  const [started, setStarted] = useState(false);
  const startButton = useRef<HTMLButtonElement>(null);
  const hasStarted = useRef(false);
  useEffect(() => { if (!started && hasStarted.current) startButton.current?.focus({preventScroll:true}); }, [started]);
  return <div className="environment-entry">
    {started ? <><div className="explorer-back"><button onClick={() => setStarted(false)}>← Back to overview</button><span>Explore RT-SAFE · NYC</span></div><ActiveEnvironmentExplorer /></>
      : <EnvironmentOverview buttonRef={startButton} onStart={() => { hasStarted.current=true; setStarted(true); }} />}
  </div>;
}
function ActiveEnvironmentExplorer() {
  const shell=useRef<HTMLDivElement>(null), canvas=useRef<HTMLCanvasElement>(null), video=useRef<HTMLVideoElement>(null);
  const fullscreenButton=useRef<HTMLButtonElement>(null);
  const renderer=useRef<Panorama|null>(null),angle=useRef<ViewAngle>({yaw:28.338,pitch:-10,fov:65});
  const rotating=useRef(false),draw=useRef<()=>void>(()=>{}),pointers=useRef(new Map<number,{x:number;y:number}>());
  const marker=useRef<HTMLButtonElement>(null),locationRef=useRef<Location|null>(null),targetRef=useRef<number[]|null>(null);
  const request=useRef(0),activeRef=useRef(false);
  const [scene,setScene]=useState<Scene|null>(null),[active,setActive]=useState("approach"),[category,setCategory]=useState<number|null>(null);
  const [mode,setMode]=useState<"explore"|"video">("explore"),[rotate,setRotate]=useState(false),[playing,setPlaying]=useState(false);
  const [ready,setReady]=useState(false),[loading,setLoading]=useState(true),[error,setError]=useState("");
  const [enabled,setEnabled]=useState(false),[detailOpen,setDetailOpen]=useState(true);
  const stop=()=>{rotating.current=false;setRotate(false);};
  const move=(id:string,focus:number|null=null)=>{stop();if(id!==active||focus!==category){request.current++;setLoading(true);}setActive(id);setCategory(focus);setDetailOpen(true);setMode("explore");video.current?.pause();};
  const reset=()=>{stop();angle.current={yaw:locationRef.current?.yaw||0,pitch:-10,fov:65};setCategory(null);draw.current();};
  useEffect(()=>{
    let owned=false;
    const changed=()=>{if(document.fullscreenElement===shell.current)owned=true;else if(owned){owned=false;fullscreenButton.current?.focus({preventScroll:true});}};
    document.addEventListener("fullscreenchange",changed);
    return()=>document.removeEventListener("fullscreenchange",changed);
  },[]);
  useEffect(()=>{
    const observer=new IntersectionObserver(entries=>{activeRef.current=entries[0].isIntersecting;if(entries[0].isIntersecting)setEnabled(true);},{rootMargin:"200px"});
    if(shell.current)observer.observe(shell.current);return()=>observer.disconnect();
  },[]);
  useEffect(()=>{
    if(!enabled)return;
    const controller=new AbortController();
    fetch("data/nyc/explorer.json",{signal:controller.signal}).then(r=>{if(!r.ok)throw Error();return r.json();}).then(setScene).catch(e=>{if(e.name!=="AbortError"){setLoading(false);setError("The scene could not load. Try reloading, or play the scene below.");}});
    return()=>controller.abort();
  },[enabled]);
  useEffect(()=>{
    if(!enabled||!canvas.current)return;
    let panorama:Panorama;
    try { panorama=new Panorama(canvas.current);renderer.current=panorama;setReady(true); }
    catch {setError("Your browser cannot display the 360° view. You can still play the scene.");setLoading(false);return;}
    const update=()=>{
      panorama.draw(angle.current);
      const point=targetRef.current,loc=locationRef.current,el=marker.current;
      if(!el)return;
      if(!point||!loc){el.hidden=true;return;}
      const [dx,dy,dz]=point.map((p,i)=>p-loc.position[i]);const y=angle.current.yaw*Math.PI/180,p=angle.current.pitch*Math.PI/180;
      const depth=dx*Math.cos(y)*Math.cos(p)+dy*Math.sin(y)*Math.cos(p)+dz*Math.sin(p);
      const right=-dx*Math.sin(y)+dy*Math.cos(y),up=-dx*Math.cos(y)*Math.sin(p)-dy*Math.sin(y)*Math.sin(p)+dz*Math.cos(p);
      const tan=Math.tan(angle.current.fov*Math.PI/360),aspect=canvas.current!.clientWidth/canvas.current!.clientHeight;
      const x=.5+.5*right/(depth*tan*aspect),v=.5-.5*up/(depth*tan);
      el.hidden=depth<=0||x<.03||x>.97||v<.06||v>.92;el.style.left=`${x*100}%`;el.style.top=`${v*100}%`;
    };
    draw.current=update;
    const resize=new ResizeObserver(update);resize.observe(canvas.current);
    let frame=0,last=0;
    const tick=(t:number)=>{if(rotating.current&&activeRef.current&&!document.hidden){angle.current.yaw+=(Math.min(t-last,50)/1000)*5;update();}last=t;frame=requestAnimationFrame(tick);};
    frame=requestAnimationFrame(tick);
    const lost=(e:Event)=>{e.preventDefault();stop();setReady(false);setError("The 360° view was interrupted. Reload the page or play the scene.");};
    const c=canvas.current;c.addEventListener("webglcontextlost",lost);
    const wheel=(e:WheelEvent)=>{if(document.activeElement!==c)return;e.preventDefault();angle.current.fov=clamp(angle.current.fov+e.deltaY*.025,35,90);update();};
    c.addEventListener("wheel",wheel,{passive:false});
    return()=>{cancelAnimationFrame(frame);resize.disconnect();c.removeEventListener("webglcontextlost",lost);c.removeEventListener("wheel",wheel);panorama.dispose();renderer.current=null;draw.current=()=>{};};
  },[enabled]);
  useEffect(()=>{
    if(!scene||!ready||!renderer.current)return;
    const generation=++request.current,loc=scene.locations.find(l=>l.id===active)!;
    locationRef.current=loc;setLoading(true);setError("");
    const point=category===null?null:scene.anchors[categories[category].anchor];targetRef.current=point;
    angle.current={yaw:loc.yaw,pitch:-10,fov:65};
    if(point){const [x,y,z]=point.map((p,i)=>p-loc.position[i]);angle.current.yaw=Math.atan2(y,x)*180/Math.PI;angle.current.pitch=clamp(Math.atan2(z,Math.hypot(x,y))*180/Math.PI,-70,70);}
    renderer.current.load(active).then(ok=>{if(ok&&generation===request.current){setLoading(false);draw.current();}}).catch(()=>{if(generation===request.current){setLoading(false);setError("This viewpoint could not load. Choose another view or play the scene.");}});
  },[scene,ready,active,category]);
  useEffect(()=>{rotating.current=rotate&&mode==="explore";},[rotate,mode]);
  const look=(dx:number,dy:number)=>{stop();angle.current.yaw-=dx*.16;angle.current.pitch=clamp(angle.current.pitch+dy*.16,-80,80);draw.current();};
  const zoom=(delta:number)=>{angle.current.fov=clamp(angle.current.fov+delta,35,90);draw.current();};
  const turn=(key:string)=>{if(key==="ArrowLeft"||key==="a")look(100,0);if(key==="ArrowRight"||key==="d")look(-100,0);if(key==="ArrowUp"||key==="w")look(0,80);if(key==="ArrowDown"||key==="s")look(0,-80);};
  const playScene=()=>{stop();setMode("video");requestAnimationFrame(()=>video.current?.play().catch(()=>{}));};
  return <div ref={shell} className="environment-explorer">
    <div className="explorer-heading"><div><span className="micro">EXPLORE RT-SAFE / NYC</span><h3>Step inside the environment.</h3></div><span className="explorer-badge">360° views</span></div>
    <div className="explorer-viewport" aria-busy={mode==="explore"&&loading}>
      <div className="explorer-panorama" style={{visibility:mode==="explore"?"visible":"hidden"}}>
        <canvas ref={canvas} tabIndex={0} aria-label="Interactive NYC panorama. Drag to look, use arrow keys to turn, plus or minus to zoom. Choose a viewpoint below to move."
          onPointerDown={e=>{if(e.button!==0)return;stop();e.currentTarget.setPointerCapture(e.pointerId);pointers.current.set(e.pointerId,{x:e.clientX,y:e.clientY});}}
          onPointerMove={e=>{const prev=pointers.current.get(e.pointerId);if(!prev)return;const next={x:e.clientX,y:e.clientY};if(pointers.current.size===2){const other=[...pointers.current.entries()].find(([id])=>id!==e.pointerId)![1];zoom((Math.hypot(prev.x-other.x,prev.y-other.y)-Math.hypot(next.x-other.x,next.y-other.y))*.15);}else look(next.x-prev.x,next.y-prev.y);pointers.current.set(e.pointerId,next);}}
          onPointerUp={e=>pointers.current.delete(e.pointerId)} onPointerCancel={e=>pointers.current.delete(e.pointerId)} onLostPointerCapture={e=>pointers.current.delete(e.pointerId)}
          onKeyDown={e=>{if(["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","w","a","s","d","+","=","-"," "].includes(e.key)){e.preventDefault();if(e.key===" ")setRotate(v=>!v);else if(e.key==="+"||e.key==="=")zoom(-5);else if(e.key==="-")zoom(5);else turn(e.key);}}} />
        {(loading||error)&&<div className="explorer-status" role="status"><strong>{error||"Opening the NYC scene…"}</strong>{error&&<button onClick={playScene}>Play scene instead</button>}</div>}
        <div className="explorer-location"><span className="explorer-dot" />{scene?.locations.find(v=>v.id===active)?.label||"The approach"}</div>
        <button hidden ref={marker} className={`explorer-hotspot category-${category}`} onClick={()=>{stop();setDetailOpen(v=>!v);}} aria-expanded={detailOpen} aria-label={category===null?"Safety event":categories[category].label}>{category===null?"":categories[category].label}<span>+</span></button>
        {category!==null&&detailOpen&&!loading&&!error&&<aside className={`explorer-insight category-${category}`}><button aria-label="Close safety detail" onClick={()=>setDetailOpen(false)}>×</button><span className="micro">{categories[category].label}</span><h4>{categories[category].title}</h4><p>{categories[category].description}</p></aside>}
        <span className="explorer-drag-hint">↔ Drag to look around</span>
      </div>
      <video ref={video} className="explorer-scene-video" style={{display:mode==="video"?"block":"none"}} muted controls playsInline preload="none" poster="media/nyc/environment.webp" aria-label="Silent NYC environment scene" onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onEnded={()=>setPlaying(false)}><source src="media/nyc/environment-clean.mp4" type="video/mp4" /></video>
      <button ref={fullscreenButton} className="explorer-fullscreen" aria-label="Toggle environment fullscreen" onClick={()=>{if(document.fullscreenElement)void document.exitFullscreen();else void shell.current?.requestFullscreen?.().catch(()=>{});}}><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6" /></svg></button>
    </div>
    <div className="explorer-controls"><div className="explorer-transport">
      <button className="explorer-primary" aria-label={mode==="video"?(playing?"Pause scene":"Play scene"):(rotate?"Pause rotation":"Auto-rotate")} onClick={()=>{if(mode==="video"){if(playing)video.current?.pause();else void video.current?.play().catch(()=>{});}else setRotate(v=>!v);}} disabled={mode==="explore"&&(loading||!!error)}>{mode==="video"?(playing?"Ⅱ Pause scene":"▶ Play scene"):(rotate?"Ⅱ Pause rotation":"↻ Auto-rotate")}</button>
      {mode==="explore"?<button onClick={playScene}>▶ Watch moving scene</button>:<button onClick={()=>{video.current?.pause();setMode("explore");requestAnimationFrame(()=>draw.current());}}>↔ Return to 360°</button>}
      {mode==="explore"&&<><button onClick={()=>zoom(-8)} aria-label="Zoom in">+</button><button onClick={()=>zoom(8)} aria-label="Zoom out">−</button><button onClick={reset}>Reset view</button></>}
    </div><span>{mode==="explore"?"Drag / arrow keys to look · + / − to zoom":"Native Unreal Engine scene · silent"}</span></div>
    <div className="explorer-destinations" aria-label="Move to a viewpoint">{(scene?.locations||[{id:"approach",label:"The approach"},{id:"hazards",label:"Along the path"},{id:"crossing",label:"The crossing"},{id:"park",label:"In the park"}]).map((loc,i)=><button key={loc.id} aria-pressed={active===loc.id&&mode==="explore"} onClick={()=>move(loc.id)}><span>0{i+1}</span>{loc.label}<span>↗</span></button>)}</div>
    <div className="explorer-topics"><span>Inspect safety events</span>{categories.map((c,i)=><button key={c.id} className={`category-${i}`} aria-pressed={category===i&&mode==="explore"} onClick={()=>move(c.view,i)}><i />{c.label} ↗</button>)}</div>
    <p className="explorer-note">Native 360° captures of our NYC demonstration scene. Click between viewpoints to explore; play the scene to see actors in motion.</p>
  </div>;
}
