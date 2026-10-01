import React from "react";
import { hydrateRoot } from "react-dom/client";
import RTsafe from "./app/rtsafe";
import "./app/globals.css";
hydrateRoot(
  document.getElementById("root")!,
  <React.StrictMode>
    <RTsafe />
  </React.StrictMode>,
);
