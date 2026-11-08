# Mean predicted-class probability on the test fold, final runs

Mean over all test visits of the probability of the predicted class, from each run's stored test predictions (`test.npz`, not committed); mean ± SD over the three seeds (1234, 2025, 7). Test visits: 3,175 / 4,174 / 4,566 at K = 10 / 20 / 30. The CEI-GNN variants are in `cei_recipe_variants/comparison.md`.

| Method | K = 10 | K = 20 | K = 30 |
|---|---|---|---|
| CEI-GNN | 0.694 ± 0.009 | 0.646 ± 0.003 | 0.609 ± 0.003 |
| GSAT | 0.751 ± 0.014 | 0.697 ± 0.018 | 0.655 ± 0.014 |
| GraphCare | 0.727 ± 0.003 | 0.650 ± 0.016 | 0.602 ± 0.009 |
| ProtGNN | 0.777 ± 0.014 | 0.734 ± 0.019 | 0.678 ± 0.019 |
