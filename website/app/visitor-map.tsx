import { useEffect, useRef } from "react";

const widgetKey = "kNtYbeOheyyz0CfffFgFNov_OGiFVuRgFJMPuBrFc-M";

export function VisitorMap() {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (window.location.hostname !== "xtrose29.github.io" || !container.current) return;
    const host = container.current;
    const script = document.createElement("script");
    script.id = "mapmyvisitors";
    script.src = `https://mapmyvisitors.com/map.js?d=${widgetKey}&cl=ffffff&w=a`;
    script.async = true;
    script.onerror = () => {
      host.textContent = "The visitor map is unavailable. View the visitor statistics below.";
    };
    host.appendChild(script);
    return () => host.replaceChildren();
  }, []);

  return (
    <section className="visitor-section" id="visitors" aria-labelledby="visitor-title">
      <div className="eyebrow">AROUND THE WORLD</div>
      <h2 id="visitor-title">Visitors</h2>
      <p>Where readers discover RT–SAFE.</p>
      <div className="visitor-map-card">
        <div ref={container} />
        <noscript>
          <a href="https://mapmyvisitors.com/web/1c8s2">
            <img src={`https://mapmyvisitors.com/map.png?d=${widgetKey}&cl=ffffff`} alt="Map of approximate RT-SAFE visitor locations" />
          </a>
        </noscript>
      </div>
      <p className="visitor-map-note">
        Locations are approximate. <a href="https://mapmyvisitors.com/web/1c8s2" target="_blank" rel="noopener noreferrer">View visitor statistics ↗</a>
      </p>
    </section>
  );
}
