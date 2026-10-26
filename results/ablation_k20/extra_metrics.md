# Secondary metrics per arm (test first; validation is the decision fold)

| arm | seeds | test balanced acc | test top-3 | validation balanced acc | validation top-3 | s/epoch | peak RSS GB |
|---|---|---|---|---|---|---|---|
| a1_full | 7,42,1234,2025,3407 | 0.7007 ± 0.0032 | 0.8961 ± 0.0024 | 0.7009 ± 0.0036 | 0.8925 ± 0.0027 | 103.3 ± 29.9 | 2.88 ± 0.43 |
| a2_mlp | 7,42,1234,2025,3407 | 0.7015 ± 0.0038 | 0.8935 ± 0.0032 | 0.7008 ± 0.0054 | 0.8852 ± 0.0045 | 158.3 ± 31.3 | 2.58 ± 0.42 |
| a3_noabsence | 7,42,1234,2025,3407 | 0.6980 ± 0.0034 | 0.8941 ± 0.0021 | 0.7015 ± 0.0042 | 0.8921 ± 0.0027 | 155.2 ± 33.9 | 2.72 ± 0.44 |
| a4_nopair | 7,42,1234,2025,3407 | 0.7047 ± 0.0026 | 0.8965 ± 0.0014 | 0.7002 ± 0.0011 | 0.8925 ± 0.0017 | 128.8 ± 34.9 | 2.83 ± 0.80 |
| a4b_nopair_noedge | 7,42,1234,2025,3407 | 0.7024 ± 0.0072 | 0.8937 ± 0.0026 | 0.7024 ± 0.0019 | 0.8887 ± 0.0029 | 98.1 ± 33.8 | 2.94 ± 0.72 |
| a5_nomp | 7,42,1234,2025,3407 | 0.7002 ± 0.0031 | 0.8937 ± 0.0024 | 0.6967 ± 0.0034 | 0.8887 ± 0.0031 | 88.1 ± 27.4 | 3.03 ± 1.04 |
| a5b_nomp_matched | 7,42,1234,2025,3407 | 0.6948 ± 0.0036 | 0.8908 ± 0.0029 | 0.6891 ± 0.0073 | 0.8831 ± 0.0045 | 113.2 ± 32.3 | 2.97 ± 0.89 |
| a6_pool | 7,42,1234,2025,3407 | 0.6921 ± 0.0074 | 0.8894 ± 0.0028 | 0.6889 ± 0.0050 | 0.8877 ± 0.0039 | 89.0 ± 14.2 | 2.37 ± 0.31 |

s/epoch and peak RSS: runs shared the Mac (several jobs at once), so they are not contention-free; the final Full runs (seeds 1234/2025/7) have no memory record.

## Test per-class F1 (mean ± SD over the arm's seeds; reported fold)

| class | a1_full | a2_mlp | a3_noabsence | a4_nopair | a4b_nopair_noedge | a5_nomp | a5b_nomp_matched | a6_pool |
|---|---|---|---|---|---|---|---|---|
| Acute kidney failure | 0.450 ± 0.016 | 0.469 ± 0.005 | 0.445 ± 0.014 | 0.468 ± 0.021 | 0.465 ± 0.017 | 0.482 ± 0.018 | 0.461 ± 0.025 | 0.452 ± 0.021 |
| Acute pharyngitis | 0.744 ± 0.021 | 0.730 ± 0.016 | 0.748 ± 0.014 | 0.753 ± 0.012 | 0.740 ± 0.020 | 0.748 ± 0.023 | 0.741 ± 0.018 | 0.732 ± 0.015 |
| Alcohol abuse with intoxication | 0.733 ± 0.007 | 0.731 ± 0.016 | 0.728 ± 0.011 | 0.737 ± 0.013 | 0.717 ± 0.018 | 0.724 ± 0.018 | 0.733 ± 0.013 | 0.724 ± 0.019 |
| Anemia | 0.675 ± 0.008 | 0.677 ± 0.006 | 0.670 ± 0.013 | 0.678 ± 0.014 | 0.677 ± 0.018 | 0.691 ± 0.016 | 0.689 ± 0.008 | 0.690 ± 0.024 |
| Back or spine pain | 0.743 ± 0.002 | 0.739 ± 0.004 | 0.742 ± 0.005 | 0.740 ± 0.010 | 0.738 ± 0.005 | 0.737 ± 0.007 | 0.741 ± 0.003 | 0.744 ± 0.007 |
| Diabetes mellitus | 0.711 ± 0.013 | 0.714 ± 0.024 | 0.711 ± 0.012 | 0.712 ± 0.011 | 0.722 ± 0.010 | 0.704 ± 0.012 | 0.696 ± 0.012 | 0.707 ± 0.028 |
| Epilepsy | 0.843 ± 0.009 | 0.838 ± 0.007 | 0.847 ± 0.014 | 0.855 ± 0.011 | 0.846 ± 0.016 | 0.843 ± 0.011 | 0.840 ± 0.008 | 0.843 ± 0.016 |
| GI bleed | 0.809 ± 0.006 | 0.813 ± 0.004 | 0.807 ± 0.006 | 0.810 ± 0.009 | 0.805 ± 0.005 | 0.816 ± 0.006 | 0.807 ± 0.004 | 0.815 ± 0.010 |
| Head injury | 0.770 ± 0.003 | 0.756 ± 0.004 | 0.772 ± 0.006 | 0.771 ± 0.008 | 0.760 ± 0.007 | 0.756 ± 0.003 | 0.761 ± 0.005 | 0.754 ± 0.004 |
| Heart failure | 0.630 ± 0.015 | 0.614 ± 0.013 | 0.624 ± 0.011 | 0.638 ± 0.021 | 0.627 ± 0.005 | 0.626 ± 0.006 | 0.614 ± 0.022 | 0.624 ± 0.031 |
| Hypokalemia | 0.580 ± 0.024 | 0.599 ± 0.018 | 0.558 ± 0.012 | 0.601 ± 0.016 | 0.586 ± 0.017 | 0.593 ± 0.016 | 0.576 ± 0.016 | 0.583 ± 0.027 |
| Limb injury or pain | 0.616 ± 0.004 | 0.608 ± 0.008 | 0.610 ± 0.004 | 0.612 ± 0.010 | 0.627 ± 0.011 | 0.611 ± 0.019 | 0.606 ± 0.008 | 0.599 ± 0.017 |
| Lower respiratory disease | 0.678 ± 0.009 | 0.665 ± 0.009 | 0.683 ± 0.006 | 0.683 ± 0.010 | 0.669 ± 0.019 | 0.676 ± 0.011 | 0.664 ± 0.011 | 0.658 ± 0.016 |
| Major depressive disorder | 0.903 ± 0.006 | 0.907 ± 0.003 | 0.906 ± 0.006 | 0.906 ± 0.002 | 0.903 ± 0.007 | 0.907 ± 0.005 | 0.910 ± 0.004 | 0.907 ± 0.011 |
| Non-ST elevation (NSTEMI) myocardial infarction | 0.746 ± 0.011 | 0.741 ± 0.014 | 0.741 ± 0.014 | 0.732 ± 0.018 | 0.713 ± 0.010 | 0.743 ± 0.008 | 0.746 ± 0.010 | 0.717 ± 0.018 |
| Sepsis | 0.406 ± 0.016 | 0.402 ± 0.034 | 0.404 ± 0.031 | 0.414 ± 0.024 | 0.418 ± 0.041 | 0.407 ± 0.021 | 0.399 ± 0.020 | 0.394 ± 0.043 |
| Skin or soft-tissue infection | 0.667 ± 0.008 | 0.663 ± 0.013 | 0.670 ± 0.004 | 0.669 ± 0.006 | 0.674 ± 0.009 | 0.673 ± 0.008 | 0.662 ± 0.006 | 0.655 ± 0.016 |
| UTI or pyelonephritis | 0.719 ± 0.004 | 0.713 ± 0.005 | 0.713 ± 0.004 | 0.713 ± 0.004 | 0.710 ± 0.005 | 0.704 ± 0.008 | 0.700 ± 0.002 | 0.703 ± 0.008 |
| Unsp intestnl obst | 0.673 ± 0.010 | 0.697 ± 0.009 | 0.675 ± 0.009 | 0.680 ± 0.010 | 0.673 ± 0.005 | 0.685 ± 0.014 | 0.687 ± 0.015 | 0.690 ± 0.047 |
| Unspecified atrial fibrillation | 0.685 ± 0.025 | 0.689 ± 0.011 | 0.675 ± 0.030 | 0.672 ± 0.019 | 0.654 ± 0.022 | 0.654 ± 0.012 | 0.650 ± 0.019 | 0.673 ± 0.020 |

## Largest test per-class F1 drops against a1_full (arm mean minus Full mean; five most negative per arm)

| arm | class: change (Full → arm) |
|---|---|
| a2_mlp | Heart failure: -0.016 (0.630 → 0.614); Acute pharyngitis: -0.014 (0.744 → 0.730); Head injury: -0.014 (0.770 → 0.756); Lower respiratory disease: -0.013 (0.678 → 0.665); Limb injury or pain: -0.007 (0.616 → 0.608) |
| a3_noabsence | Hypokalemia: -0.022 (0.580 → 0.558); Unspecified atrial fibrillation: -0.010 (0.685 → 0.675); UTI or pyelonephritis: -0.006 (0.719 → 0.713); Heart failure: -0.006 (0.630 → 0.624); Limb injury or pain: -0.006 (0.616 → 0.610) |
| a4_nopair | Non-ST elevation (NSTEMI) myocardial infarction: -0.014 (0.746 → 0.732); Unspecified atrial fibrillation: -0.013 (0.685 → 0.672); UTI or pyelonephritis: -0.006 (0.719 → 0.713); Limb injury or pain: -0.003 (0.616 → 0.612); Back or spine pain: -0.003 (0.743 → 0.740) |
| a4b_nopair_noedge | Non-ST elevation (NSTEMI) myocardial infarction: -0.033 (0.746 → 0.713); Unspecified atrial fibrillation: -0.031 (0.685 → 0.654); Alcohol abuse with intoxication: -0.016 (0.733 → 0.717); Head injury: -0.010 (0.770 → 0.760); Lower respiratory disease: -0.010 (0.678 → 0.669) |
| a5_nomp | Unspecified atrial fibrillation: -0.031 (0.685 → 0.654); UTI or pyelonephritis: -0.015 (0.719 → 0.704); Head injury: -0.014 (0.770 → 0.756); Alcohol abuse with intoxication: -0.009 (0.733 → 0.724); Diabetes mellitus: -0.007 (0.711 → 0.704) |
| a5b_nomp_matched | Unspecified atrial fibrillation: -0.035 (0.685 → 0.650); UTI or pyelonephritis: -0.019 (0.719 → 0.700); Heart failure: -0.016 (0.630 → 0.614); Diabetes mellitus: -0.016 (0.711 → 0.696); Lower respiratory disease: -0.014 (0.678 → 0.664) |
| a6_pool | Non-ST elevation (NSTEMI) myocardial infarction: -0.029 (0.746 → 0.717); Lower respiratory disease: -0.020 (0.678 → 0.658); Head injury: -0.016 (0.770 → 0.754); Limb injury or pain: -0.016 (0.616 → 0.599); UTI or pyelonephritis: -0.016 (0.719 → 0.703) |

## Validation per-class F1 (mean over seeds; decision fold)

| class | a1_full | a2_mlp | a3_noabsence | a4_nopair | a4b_nopair_noedge | a5_nomp | a5b_nomp_matched | a6_pool |
|---|---|---|---|---|---|---|---|---|
| Acute kidney failure | 0.506 | 0.515 | 0.511 | 0.522 | 0.522 | 0.500 | 0.497 | 0.480 |
| Acute pharyngitis | 0.765 | 0.779 | 0.772 | 0.766 | 0.750 | 0.771 | 0.774 | 0.767 |
| Alcohol abuse with intoxication | 0.756 | 0.747 | 0.752 | 0.761 | 0.765 | 0.740 | 0.760 | 0.754 |
| Anemia | 0.659 | 0.654 | 0.651 | 0.658 | 0.659 | 0.659 | 0.655 | 0.623 |
| Back or spine pain | 0.737 | 0.734 | 0.736 | 0.736 | 0.734 | 0.734 | 0.732 | 0.743 |
| Diabetes mellitus | 0.700 | 0.711 | 0.701 | 0.697 | 0.700 | 0.681 | 0.669 | 0.682 |
| Epilepsy | 0.816 | 0.821 | 0.821 | 0.818 | 0.809 | 0.816 | 0.817 | 0.829 |
| GI bleed | 0.825 | 0.819 | 0.822 | 0.822 | 0.817 | 0.822 | 0.828 | 0.812 |
| Head injury | 0.710 | 0.714 | 0.711 | 0.709 | 0.709 | 0.710 | 0.708 | 0.711 |
| Heart failure | 0.610 | 0.601 | 0.605 | 0.599 | 0.611 | 0.603 | 0.591 | 0.598 |
| Hypokalemia | 0.602 | 0.580 | 0.599 | 0.600 | 0.591 | 0.601 | 0.588 | 0.576 |
| Limb injury or pain | 0.596 | 0.600 | 0.604 | 0.592 | 0.607 | 0.586 | 0.584 | 0.589 |
| Lower respiratory disease | 0.629 | 0.625 | 0.630 | 0.640 | 0.636 | 0.633 | 0.628 | 0.636 |
| Major depressive disorder | 0.906 | 0.905 | 0.906 | 0.907 | 0.904 | 0.896 | 0.901 | 0.897 |
| Non-ST elevation (NSTEMI) myocardial infarction | 0.783 | 0.774 | 0.779 | 0.768 | 0.762 | 0.783 | 0.756 | 0.792 |
| Sepsis | 0.368 | 0.343 | 0.377 | 0.373 | 0.375 | 0.363 | 0.346 | 0.378 |
| Skin or soft-tissue infection | 0.630 | 0.641 | 0.633 | 0.627 | 0.630 | 0.631 | 0.621 | 0.625 |
| UTI or pyelonephritis | 0.721 | 0.715 | 0.721 | 0.719 | 0.721 | 0.714 | 0.705 | 0.697 |
| Unsp intestnl obst | 0.732 | 0.736 | 0.733 | 0.731 | 0.724 | 0.729 | 0.718 | 0.734 |
| Unspecified atrial fibrillation | 0.669 | 0.676 | 0.664 | 0.667 | 0.665 | 0.657 | 0.643 | 0.690 |
