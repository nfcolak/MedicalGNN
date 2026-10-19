# Edge-direction pilot (10k training rows)

Protocol fair_v10t with the `train_limit` override (10000 training rows), K=20 (20 classes), seed 1234, one run per arm. Decisions (checkpoint, HPO, baseline choice) use VALIDATION; every reported result is on TEST (scored once per run at the validation-selected checkpoint). Test 4174 rows / 3151 patients; validation (decision fold) 4183 rows / 3142 patients.
Difference = first minus second in macro-F1; [95% patient-bootstrap interval, B=2000, one resample shared by all arms, rng 20261016]; P = share of resamples with a positive difference. One seed: the intervals are conditional on that seed and on the fixed fold.

## Runs

| run | edge direction | parameters | selected epoch | test macro-F1 | test accuracy | validation macro-F1 (decision fold) | validation accuracy |
|---|---|---|---|---|---|---|---|
| cei_mp2 | forward | 211296 | 26 | 0.6633 | 0.6799 | 0.6591 | 0.6730 |
| graphcare_forward | forward | 134396 | 39 | 0.6437 | 0.6593 | 0.6459 | 0.6603 |
| graphcare_bidirectional | bidirectional | 140796 | 29 | 0.6288 | 0.6478 | 0.6413 | 0.6543 |
| gsat_forward | forward | 211576 | 31 | 0.6316 | 0.6416 | 0.6320 | 0.6462 |
| gsat_bidirectional | bidirectional | 217976 | 36 | 0.6341 | 0.6538 | 0.6353 | 0.6493 |
| protgnn_forward | forward | 420784 | 13 | 0.6333 | 0.6543 | 0.6356 | 0.6512 |
| protgnn_bidirectional | bidirectional | 427184 | 29 | 0.6391 | 0.6564 | 0.6410 | 0.6538 |

## Bidirectional minus forward, per baseline

| pair | test a / b | d [CI] | P(d>0) | validation (decision fold) a / b | d [CI] | P(d>0) |
|---|---|---|---|---|---|---|
| GraphCare bidirectional - forward | 0.6288 / 0.6437 | -0.0149 [-0.0264, -0.0038] | 0.004 | 0.6413 / 0.6459 | -0.0046 [-0.0156, +0.0065] | 0.233 |
| GSAT bidirectional - forward | 0.6341 / 0.6316 | +0.0025 [-0.0100, +0.0158] | 0.666 | 0.6353 / 0.6320 | +0.0033 [-0.0093, +0.0159] | 0.699 |
| ProtGNN bidirectional - forward | 0.6391 / 0.6333 | +0.0058 [-0.0083, +0.0197] | 0.809 | 0.6410 / 0.6356 | +0.0053 [-0.0080, +0.0177] | 0.782 |

## CEI-GNN mp2 minus each baseline arm

| pair | test a / b | d [CI] | P(d>0) | validation (decision fold) a / b | d [CI] | P(d>0) |
|---|---|---|---|---|---|---|
| CEI-GNN mp2 - GraphCare bidirectional | 0.6633 / 0.6288 | +0.0344 [+0.0226, +0.0464] | 1.000 | 0.6591 / 0.6413 | +0.0178 [+0.0051, +0.0310] | 0.998 |
| CEI-GNN mp2 - GSAT bidirectional | 0.6633 / 0.6341 | +0.0291 [+0.0163, +0.0412] | 1.000 | 0.6591 / 0.6353 | +0.0238 [+0.0109, +0.0367] | 1.000 |
| CEI-GNN mp2 - ProtGNN bidirectional | 0.6633 / 0.6391 | +0.0242 [+0.0114, +0.0372] | 1.000 | 0.6591 / 0.6410 | +0.0181 [+0.0057, +0.0312] | 0.998 |
| CEI-GNN mp2 - GraphCare forward | 0.6633 / 0.6437 | +0.0196 [+0.0080, +0.0311] | 1.000 | 0.6591 / 0.6459 | +0.0132 [+0.0009, +0.0261] | 0.982 |
| CEI-GNN mp2 - GSAT forward | 0.6633 / 0.6316 | +0.0316 [+0.0181, +0.0456] | 1.000 | 0.6591 / 0.6320 | +0.0271 [+0.0137, +0.0411] | 1.000 |
| CEI-GNN mp2 - ProtGNN forward | 0.6633 / 0.6333 | +0.0300 [+0.0171, +0.0432] | 1.000 | 0.6591 / 0.6356 | +0.0235 [+0.0110, +0.0364] | 1.000 |

## Binding differences between the two arms of a baseline

- GraphCare: active_parameter_count, edge_dim, edge_direction, edge_feature_layout, meta_relations, method_config.architecture.edge_dim, method_config.architecture.num_relations, method_config.architecture.num_triples, method_config.architecture.parameter_count, num_meta_relations, num_relations, parameter_count, selected_test.epoch, selected_test.epoch_index, selected_test.prediction_sha256, selected_validation.epoch, selected_validation.epoch_index, selected_validation.metric_value, selected_validation.prediction_sha256
- GSAT: active_parameter_count, edge_dim, edge_direction, edge_feature_layout, meta_relations, method_config.architecture.edge_dim, method_config.architecture.num_relations, method_config.architecture.num_triples, method_config.architecture.parameter_count, num_meta_relations, num_relations, parameter_count, selected_test.epoch, selected_test.epoch_index, selected_test.prediction_sha256, selected_validation.epoch, selected_validation.epoch_index, selected_validation.metric_value, selected_validation.prediction_sha256
- ProtGNN: active_parameter_count, edge_dim, edge_direction, edge_feature_layout, meta_relations, method_config.architecture.edge_dim, method_config.architecture.joint_active_parameter_count, method_config.architecture.num_relations, method_config.architecture.num_triples, method_config.architecture.parameter_count, method_config.architecture.warmup_active_parameter_count, num_meta_relations, num_relations, parameter_count, selected_test.epoch, selected_test.epoch_index, selected_test.prediction_sha256, selected_validation.epoch, selected_validation.epoch_index, selected_validation.metric_value, selected_validation.prediction_sha256
