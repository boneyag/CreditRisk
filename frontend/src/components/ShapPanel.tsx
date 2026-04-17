import type { ExplanationItem } from "../types";

type ShapPanelProps = {
  title: string;
  subtitle: string;
  items: ExplanationItem[];
  limit?: number;
};

export default function ShapPanel({ title, subtitle, items, limit = 10 }: ShapPanelProps) {
  const topItems = items.slice(0, limit);
  const max = Math.max(...topItems.map((item) => item.abs_shap_value), 0.0001);

  return (
    <section className="panel">
      <div className="panel-head">
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>

      <div className="bar-list">
        {topItems.map((item) => {
          const width = `${Math.max((item.abs_shap_value / max) * 100, 2)}%`;
          const positive = item.shap_value >= 0;
          return (
            <div className="bar-row" key={`${item.feature}-${item.shap_value}`}>
              <div className="bar-meta">
                <span className="feature-name" title={item.feature}>
                  {item.feature}
                </span>
                <span className={`feature-value ${positive ? "up" : "down"}`}>
                  {positive ? "+" : ""}
                  {item.shap_value.toFixed(4)}
                </span>
              </div>
              <div className="bar-track" role="img" aria-label={`${item.feature} SHAP contribution`}>
                <div className={`bar-fill ${positive ? "up" : "down"}`} style={{ width }} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
