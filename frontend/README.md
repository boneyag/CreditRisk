# CreditRisk UI

Frontend for model inference and explanation review.

## Stack
- React 18 + TypeScript + Vite
- TanStack Query for API request state
- React Hook Form + Zod for payload validation

## Page Blueprint
- `src/App.tsx`: page shell + mutation wiring + result panels
- `src/components/InferenceForm.tsx`: typed form mapped to API payload fields
- `src/components/ShapPanel.tsx`: ranked horizontal contribution bars
- `src/api.ts`: API client for `/explain`
- `src/types.ts`: request/response contracts
- `src/schema.ts`: validation and defaults

## Data Flow
1. User submits inference form.
2. Form values are validated by Zod.
3. UI posts payload to `POST /explain`.
4. API response is rendered in three blocks:
   - prediction snapshot
   - grouped SHAP contributions (`feature_explanations`)
   - transformed-feature SHAP contributions (`transformed_feature_explanations`)

## Local Run
```bash
npm install
npm run dev
```

## API Base URL
Default is `http://localhost:8000`.

To override:
```bash
VITE_API_BASE=http://localhost:8000 npm run dev
```
