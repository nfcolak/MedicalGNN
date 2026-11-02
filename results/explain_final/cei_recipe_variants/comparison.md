# CEI-GNN recipe variants: comparison with the final model

Each K is a separately trained model set; mean over seeds 1234 / 2025 / 7, 500 test patients per K. "score" = `deletion_mean_change_vs_random` (higher is better). Test macro-F1, accuracy and mean predicted-class probability are over all test visits. Flip % = share of visits where the predicted class changes when all relation edges are removed (trained models, inference only). Seed agreement and weight randomisation are for the built-in ledger (lift over chance; Spearman, low is good).

## K = 10

| model | LS | WD | DE | macro-F1 | acc | conf | IG | Grad | GNNExp | ledger | mass90 | flip % | seed lift | w-rand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEI-GNN final | x | x | x | 0.761 | 0.754 | 0.69 | 0.248 | 0.244 | 0.205 | 0.270 | 0.70 | 27 | 0.45 | -0.03 |
| weight decay removed | x | - | x | 0.763 | 0.756 | 0.69 | 0.244 | 0.245 | 0.204 | 0.273 | 0.70 | 27 | 0.46 | -0.01 |
| DropEdge removed | x | x | - | 0.764 | 0.758 | 0.70 | 0.255 | 0.246 | 0.208 | 0.257 | 0.71 | 46 | 0.45 | -0.03 |
| label smoothing removed | - | x | x | 0.764 | 0.759 | 0.78 | 0.289 | 0.272 | 0.219 | 0.328 | 0.66 | 28 | 0.47 | +0.02 |
| label smoothing + weight decay removed | - | - | x | 0.763 | 0.758 | 0.78 | 0.285 | 0.271 | 0.218 | 0.330 | 0.66 | 28 | 0.47 | +0.02 |
| all three removed | - | - | - | 0.763 | 0.759 | 0.78 | 0.288 | 0.272 | 0.221 | 0.320 | 0.66 | 43 | 0.46 | +0.02 |

## K = 20

| model | LS | WD | DE | macro-F1 | acc | conf | IG | Grad | GNNExp | ledger | mass90 | flip % | seed lift | w-rand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEI-GNN final | x | x | x | 0.689 | 0.704 | 0.65 | 0.259 | 0.254 | 0.232 | 0.275 | 0.75 | 26 | 0.48 | -0.09 |
| weight decay removed | x | - | x | 0.690 | 0.704 | 0.65 | 0.259 | 0.253 | 0.229 | 0.277 | 0.74 | 27 | 0.47 | -0.10 |
| DropEdge removed | x | x | - | 0.690 | 0.705 | 0.65 | 0.261 | 0.255 | 0.225 | 0.276 | 0.74 | 64 | 0.49 | -0.11 |
| label smoothing removed | - | x | x | 0.680 | 0.700 | 0.73 | 0.295 | 0.287 | 0.247 | 0.324 | 0.71 | 30 | 0.48 | -0.09 |
| label smoothing + weight decay removed | - | - | x | 0.679 | 0.697 | 0.73 | 0.292 | 0.287 | 0.242 | 0.327 | 0.70 | 29 | 0.47 | -0.06 |
| all three removed | - | - | - | 0.682 | 0.700 | 0.73 | 0.294 | 0.286 | 0.241 | 0.323 | 0.71 | 51 | 0.48 | -0.05 |

## K = 30

| model | LS | WD | DE | macro-F1 | acc | conf | IG | Grad | GNNExp | ledger | mass90 | flip % | seed lift | w-rand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEI-GNN final | x | x | x | 0.578 | 0.666 | 0.61 | 0.262 | 0.259 | 0.220 | 0.275 | 0.72 | 33 | 0.53 | -0.06 |
| weight decay removed | x | - | x | 0.573 | 0.665 | 0.61 | 0.264 | 0.261 | 0.224 | 0.277 | 0.72 | 31 | 0.53 | -0.06 |
| DropEdge removed | x | x | - | 0.573 | 0.665 | 0.61 | 0.264 | 0.261 | 0.223 | 0.276 | 0.73 | 57 | 0.53 | -0.08 |
| label smoothing removed | - | x | x | 0.570 | 0.663 | 0.69 | 0.294 | 0.290 | 0.243 | 0.319 | 0.70 | 35 | 0.54 | -0.07 |
| label smoothing + weight decay removed | - | - | x | 0.573 | 0.663 | 0.69 | 0.295 | 0.293 | 0.247 | 0.321 | 0.70 | 36 | 0.53 | -0.09 |
| all three removed | - | - | - | 0.566 | 0.660 | 0.69 | 0.290 | 0.287 | 0.240 | 0.319 | 0.69 | 55 | 0.53 | -0.07 |
