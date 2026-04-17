import { useMutation } from "@tanstack/react-query";

import { explainInference } from "./api";
import InferenceForm from "./components/InferenceForm";
import ShapPanel from "./components/ShapPanel";
import type { InferenceFormValues } from "./schema";

function formatProbability(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatDecision(prediction: number) {
  return prediction === 1 ? "Approved / Low Risk" : "Rejected / High Risk";
}

export default function App() {
  const explainMutation = useMutation({
    mutationFn: explainInference,
  });

  const result = explainMutation.data;

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="kicker">CreditRisk Service UI</p>
        <h1>Inference Console</h1>
        <p>
          Validate inputs, run model inference, and inspect SHAP attributions grouped by source feature and by
          transformed pipeline columns.
        </p>
      </header>

      <main className="layout">
        <InferenceForm isLoading={explainMutation.isPending} onSubmit={(values: InferenceFormValues) => explainMutation.mutate(values)} />

        <section className="results-column">
          <section className="panel prediction-panel">
            <div className="panel-head">
              <h2>Prediction Snapshot</h2>
              <p>Latest score and model metadata from the API response.</p>
            </div>

            {result ? (
              <div className="prediction-grid">
                <article>
                  <h4>Decision</h4>
                  <p className={`big-number ${result.pred === 1 ? "good" : "bad"}`}>{formatDecision(result.pred)}</p>
                </article>
                <article>
                  <h4>Approval Probability</h4>
                  <p className="big-number">{formatProbability(result.proba)}</p>
                </article>
                <article>
                  <h4>Base Value</h4>
                  <p>{result.expected_value.toFixed(4)}</p>
                </article>
                <article>
                  <h4>Model Version</h4>
                  <p>{result.model_version}</p>
                </article>
              </div>
            ) : (
              <p className="empty-state">Submit a request to see prediction and SHAP outputs.</p>
            )}

            {explainMutation.error ? <p className="error-inline">{explainMutation.error.message}</p> : null}
          </section>

          {result ? (
            <>
              <ShapPanel
                title="Grouped Explanations"
                subtitle="One-hot columns are rolled up into their source fields."
                items={result.feature_explanations}
                limit={12}
              />
              <ShapPanel
                title="Transformed Explanations"
                subtitle={`Raw transformed columns from the preprocessor (${result.transformed_feature_count} total).`}
                items={result.transformed_feature_explanations}
                limit={12}
              />
            </>
          ) : null}
        </section>
      </main>
    </div>
  );
}
