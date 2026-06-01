// Finally - this is the flow. The last ChargeOptimizationBudgetMessage includes the last_charge_id at the time of triggering it
// ChargeOptimizationBudgetHandler
//           │
//           ▼
//   TransactionCreatedEvent (OptimizationCharged)
//           │
//           ▼
// TransactionCreatedHandler
//           │
//           ▼
//     Seed/Breed Job
//           │
//           ▼
//   Genotypes Evaluated
//           │
//           ▼
// GenotypeEvaluatedHandler ──────────────────────┐
//           │                                    │
//           │ (when ready to charge)             │ (loop)
//           ▼                                    │
// ChargeOptimizationBudgetMessage ───────────────┘
