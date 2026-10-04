const marks: Record<string, string> = {
  Astra: "openai", Sol: "openai", Fable: "claude", Sonnet: "claude",
  Gemini: "gemini", DeepSeek: "deepseek", Grok: "grok", Inkling: "thinking-machines",
};
export function ModelLogo({ name }: { name: string }) {
  return <img className="model-logo" src={`media/logos/${marks[name]}.png`} width="32" height="32" alt="" aria-hidden="true" />;
}
