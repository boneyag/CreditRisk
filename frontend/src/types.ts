export type PredictionRequest = {
  age: number;
  occupation_status: "Employed" | "Student" | "Self-Employed";
  years_employed: number;
  annual_income: number;
  credit_score: number;
  credit_history_years: number;
  savings_assets: number;
  current_debt: number;
  defaults_on_file: number;
  delinquencies_last_2yrs: number;
  derogatory_marks: number;
  product_type: "Credit Card" | "Personal Loan" | "Line of Credit";
  loan_intent: "Business" | "Home Improvement" | "Debt Consolidation" | "Education" | "Personal" | "Medical";
  loan_amount: number;
  interest_rate: number;
  debt_to_income_ratio: number;
  loan_to_income_ratio: number;
  payment_to_income_ratio: number;
};

export type ExplanationItem = {
  feature: string;
  shap_value: number;
  abs_shap_value: number;
};

export type ExplainResponse = {
  pred: number;
  proba: number;
  model_version: string;
  expected_value: number;
  feature_explanations: ExplanationItem[];
  transformed_feature_explanations: ExplanationItem[];
  transformed_feature_count: number;
};
