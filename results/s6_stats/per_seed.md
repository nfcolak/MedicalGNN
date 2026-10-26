# Per-seed results

Final runs, protocol fair_v10t; seeds 1234, 2025, 7; methods CEI-GNN mp2, GraphCare-style (adapted), GSAT, ProtGNN. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint).

## K=10

test: 3175 rows, 2497 patients / validation: 3165 rows, 2473 patients

| method | seed | test macro-F1 | test acc | test bal. acc | val macro-F1 (decision fold) | val acc (decision fold) | selected epoch (validation) |
|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 1234 | 0.7529 | 0.7446 | 0.7540 | 0.7444 | 0.7393 | 16 |
| CEI-GNN mp2 | 2025 | 0.7707 | 0.7641 | 0.7742 | 0.7487 | 0.7441 | 40 |
| CEI-GNN mp2 | 7 | 0.7600 | 0.7543 | 0.7631 | 0.7422 | 0.7397 | 31 |
| GraphCare-style (adapted) | 1234 | 0.7665 | 0.7572 | 0.7681 | 0.7339 | 0.7321 | 34 |
| GraphCare-style (adapted) | 2025 | 0.7605 | 0.7534 | 0.7632 | 0.7296 | 0.7283 | 30 |
| GraphCare-style (adapted) | 7 | 0.7616 | 0.7524 | 0.7659 | 0.7361 | 0.7343 | 28 |
| GSAT | 1234 | 0.7462 | 0.7446 | 0.7520 | 0.7257 | 0.7251 | 21 |
| GSAT | 2025 | 0.7524 | 0.7474 | 0.7570 | 0.7250 | 0.7251 | 29 |
| GSAT | 7 | 0.7541 | 0.7474 | 0.7582 | 0.7278 | 0.7270 | 33 |
| ProtGNN | 1234 | 0.7456 | 0.7402 | 0.7532 | 0.7313 | 0.7314 | 21 |
| ProtGNN | 2025 | 0.7497 | 0.7436 | 0.7622 | 0.7324 | 0.7299 | 27 |
| ProtGNN | 7 | 0.7577 | 0.7515 | 0.7655 | 0.7378 | 0.7374 | 17 |

## K=20

test: 4174 rows, 3151 patients / validation: 4183 rows, 3142 patients

| method | seed | test macro-F1 | test acc | test bal. acc | val macro-F1 (decision fold) | val acc (decision fold) | selected epoch (validation) |
|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 1234 | 0.6889 | 0.7036 | 0.7024 | 0.6843 | 0.6950 | 30 |
| CEI-GNN mp2 | 2025 | 0.6883 | 0.7036 | 0.6976 | 0.6829 | 0.6969 | 40 |
| CEI-GNN mp2 | 7 | 0.6909 | 0.7056 | 0.7046 | 0.6904 | 0.7016 | 38 |
| GraphCare-style (adapted) | 1234 | 0.6724 | 0.6900 | 0.6792 | 0.6759 | 0.6895 | 38 |
| GraphCare-style (adapted) | 2025 | 0.6666 | 0.6818 | 0.6771 | 0.6725 | 0.6844 | 19 |
| GraphCare-style (adapted) | 7 | 0.6751 | 0.6912 | 0.6871 | 0.6743 | 0.6859 | 38 |
| GSAT | 1234 | 0.6737 | 0.6897 | 0.6888 | 0.6744 | 0.6852 | 32 |
| GSAT | 2025 | 0.6657 | 0.6809 | 0.6712 | 0.6723 | 0.6835 | 20 |
| GSAT | 7 | 0.6683 | 0.6838 | 0.6840 | 0.6687 | 0.6840 | 37 |
| ProtGNN | 1234 | 0.6646 | 0.6852 | 0.6749 | 0.6759 | 0.6828 | 8 |
| ProtGNN | 2025 | 0.6681 | 0.6883 | 0.6808 | 0.6697 | 0.6847 | 30 |
| ProtGNN | 7 | 0.6637 | 0.6818 | 0.6706 | 0.6774 | 0.6875 | 31 |

## K=30

test: 4566 rows, 3404 patients / validation: 4534 rows, 3388 patients

| method | seed | test macro-F1 | test acc | test bal. acc | val macro-F1 (decision fold) | val acc (decision fold) | selected epoch (validation) |
|---|---|---|---|---|---|---|---|
| CEI-GNN mp2 | 1234 | 0.5754 | 0.6658 | 0.6220 | 0.5749 | 0.6599 | 38 |
| CEI-GNN mp2 | 2025 | 0.5768 | 0.6651 | 0.6315 | 0.5677 | 0.6584 | 27 |
| CEI-GNN mp2 | 7 | 0.5822 | 0.6675 | 0.6248 | 0.5697 | 0.6546 | 25 |
| GraphCare-style (adapted) | 1234 | 0.5491 | 0.6402 | 0.5820 | 0.5506 | 0.6449 | 37 |
| GraphCare-style (adapted) | 2025 | 0.5453 | 0.6456 | 0.5730 | 0.5498 | 0.6453 | 34 |
| GraphCare-style (adapted) | 7 | 0.5458 | 0.6483 | 0.5700 | 0.5469 | 0.6453 | 20 |
| GSAT | 1234 | 0.5637 | 0.6483 | 0.6010 | 0.5568 | 0.6434 | 34 |
| GSAT | 2025 | 0.5584 | 0.6553 | 0.5912 | 0.5522 | 0.6458 | 29 |
| GSAT | 7 | 0.5489 | 0.6413 | 0.5970 | 0.5552 | 0.6445 | 38 |
| ProtGNN | 1234 | 0.5591 | 0.6511 | 0.5895 | 0.5731 | 0.6484 | 24 |
| ProtGNN | 2025 | 0.5562 | 0.6509 | 0.5979 | 0.5605 | 0.6495 | 30 |
| ProtGNN | 7 | 0.5546 | 0.6395 | 0.5990 | 0.5559 | 0.6434 | 14 |

