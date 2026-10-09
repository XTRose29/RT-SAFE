import { useEffect, useState } from "react";

export function VisitorTracking() {
  const [enabled, setEnabled] = useState(false);
  useEffect(() => {
    setEnabled(window.location.hostname === "xtrose29.github.io");
  }, []);
  if (!enabled) return null;
  return <img
    src="https://mapmyvisitors.com/map.png?d=kNtYbeOheyyz0CfffFgFNov_OGiFVuRgFJMPuBrFc-M&cl=ffffff"
    width="1" height="1" alt="" aria-hidden="true"
    style={{ position: "absolute", width: 1, height: 1, pointerEvents: "none" }}
  />;
}
