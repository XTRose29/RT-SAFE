import { renderToString } from "react-dom/server";
import RTsafe from "./app/rtsafe";
export function render() {
  return renderToString(<RTsafe />);
}
