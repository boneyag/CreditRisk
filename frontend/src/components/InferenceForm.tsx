import { useMemo } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

import { defaultValues, inferenceSchema, type InferenceFormValues } from "../schema";

type InferenceFormProps = {
  onSubmit: (values: InferenceFormValues) => void;
  isLoading: boolean;
};

type FieldConfig = {
  name: keyof InferenceFormValues;
  label: string;
  type?: "number" | "text";
  step?: string;
  options?: Array<{ label: string; value: string }>;
};

const fields: FieldConfig[] = [
  { name: "age", label: "Age", type: "number" },
  {
    name: "occupation_status",
    label: "Occupation Status",
    options: [
      { label: "Employed", value: "Employed" },
      { label: "Student", value: "Student" },
      { label: "Self-Employed", value: "Self-Employed" },
    ],
  },
  { name: "years_employed", label: "Years Employed", type: "number" },
  { name: "annual_income", label: "Annual Income", type: "number", step: "100" },
  { name: "credit_score", label: "Credit Score", type: "number" },
  { name: "credit_history_years", label: "Credit History Years", type: "number" },
  { name: "savings_assets", label: "Savings Assets", type: "number", step: "100" },
  { name: "current_debt", label: "Current Debt", type: "number", step: "100" },
  { name: "defaults_on_file", label: "Defaults On File", type: "number" },
  { name: "delinquencies_last_2yrs", label: "Delinquencies Last 2Y", type: "number" },
  { name: "derogatory_marks", label: "Derogatory Marks", type: "number" },
  {
    name: "product_type",
    label: "Product Type",
    options: [
      { label: "Credit Card", value: "Credit Card" },
      { label: "Personal Loan", value: "Personal Loan" },
      { label: "Line of Credit", value: "Line of Credit" },
    ],
  },
  {
    name: "loan_intent",
    label: "Loan Intent",
    options: [
      { label: "Business", value: "Business" },
      { label: "Home Improvement", value: "Home Improvement" },
      { label: "Debt Consolidation", value: "Debt Consolidation" },
      { label: "Education", value: "Education" },
      { label: "Personal", value: "Personal" },
      { label: "Medical", value: "Medical" },
    ],
  },
  { name: "loan_amount", label: "Loan Amount", type: "number", step: "100" },
  { name: "interest_rate", label: "Interest Rate", type: "number", step: "0.01" },
  { name: "debt_to_income_ratio", label: "Debt/Income Ratio", type: "number", step: "0.01" },
  { name: "loan_to_income_ratio", label: "Loan/Income Ratio", type: "number", step: "0.01" },
  { name: "payment_to_income_ratio", label: "Payment/Income Ratio", type: "number", step: "0.01" },
];

export default function InferenceForm({ onSubmit, isLoading }: InferenceFormProps) {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<InferenceFormValues>({
    resolver: zodResolver(inferenceSchema),
    defaultValues,
  });

  const errorCount = useMemo(() => Object.keys(errors).length, [errors]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Inference Request</h2>
        <p>Submit borrower attributes to generate prediction + SHAP explanations.</p>
      </div>

      <form className="form-grid" onSubmit={handleSubmit(onSubmit)}>
        {fields.map((field) => {
          const error = errors[field.name]?.message;
          const id = `field-${field.name}`;
          return (
            <label className="field" key={field.name} htmlFor={id}>
              <span>{field.label}</span>
              {field.options ? (
                <select id={id} {...register(field.name)}>
                  {field.options.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input id={id} type={field.type || "number"} step={field.step} {...register(field.name)} />
              )}
              {error ? <small className="error">{error}</small> : null}
            </label>
          );
        })}

        <div className="form-actions">
          <button className="btn-primary" type="submit" disabled={isLoading}>
            {isLoading ? "Scoring..." : "Run Inference"}
          </button>
          {errorCount > 0 ? <p className="error-inline">{errorCount} fields need attention.</p> : null}
        </div>
      </form>
    </section>
  );
}
