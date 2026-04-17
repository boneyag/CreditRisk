import { z } from "zod";

export const inferenceSchema = z.object({
  age: z.coerce.number().int().min(18).max(100),
  occupation_status: z.enum(["Employed", "Student", "Self-Employed"]),
  years_employed: z.coerce.number().min(0).max(60),
  annual_income: z.coerce.number().min(0),
  credit_score: z.coerce.number().min(250).max(900),
  credit_history_years: z.coerce.number().min(0).max(80),
  savings_assets: z.coerce.number().min(0),
  current_debt: z.coerce.number().min(0),
  defaults_on_file: z.coerce.number().int().min(0).max(20),
  delinquencies_last_2yrs: z.coerce.number().int().min(0).max(50),
  derogatory_marks: z.coerce.number().int().min(0).max(50),
  product_type: z.enum(["Credit Card", "Personal Loan", "Line of Credit"]),
  loan_intent: z.enum(["Business", "Home Improvement", "Debt Consolidation", "Education", "Personal", "Medical"]),
  loan_amount: z.coerce.number().min(0),
  interest_rate: z.coerce.number().min(0).max(1),
  debt_to_income_ratio: z.coerce.number().min(0).max(1),
  loan_to_income_ratio: z.coerce.number().min(0).max(2),
  payment_to_income_ratio: z.coerce.number().min(0).max(1),
});

export type InferenceFormValues = z.infer<typeof inferenceSchema>;

export const defaultValues: InferenceFormValues = {
  age: 42,
  occupation_status: "Employed",
  years_employed: 10,
  annual_income: 85000,
  credit_score: 720,
  credit_history_years: 12,
  savings_assets: 15000,
  current_debt: 12000,
  defaults_on_file: 0,
  delinquencies_last_2yrs: 1,
  derogatory_marks: 0,
  product_type: "Personal Loan",
  loan_intent: "Debt Consolidation",
  loan_amount: 15000,
  interest_rate: 0.12,
  debt_to_income_ratio: 0.18,
  loan_to_income_ratio: 0.35,
  payment_to_income_ratio: 0.08,
};
