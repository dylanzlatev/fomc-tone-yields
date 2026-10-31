# Agreement summary: gold

Metrics use each item's primary aspect (largest |direction|, ties -> first listed), on items labelled by all labellers. Percentages are 0-100.

## Overall
| n_items | n_labelers | function_fleiss_kappa | function_all_agree_pct | topic_n_items | topic_all_agree_pct | topic_fleiss_kappa | direction_all_exact_pct | direction_alpha_interval | direction_subst_alpha_interval | full_match_all_pct | n_items_labelled_by_any |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 350 | 3 | 0.959 | 96.000 | 224 | 95.536 | 0.958 | 87.714 | 0.949 | 0.950 | 83.714 | 350 |

## Pairwise
| labeler_a | labeler_b | n_items | function_pct | function_kappa | topic_n_items | topic_pct | topic_kappa | direction_exact_pct | direction_within1_pct | direction_sign_pct | direction_qwk | direction_subst_n_items | direction_subst_exact_pct | direction_subst_qwk | full_match_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus-low | opus-med | 350 | 98.286 | 0.973 | 225 | 99.111 | 0.987 | 94.571 | 99.714 | 95.714 | 0.968 | 332 | 94.277 | 0.968 | 93.429 |
| opus-low | sonnet | 350 | 96.286 | 0.942 | 225 | 95.556 | 0.937 | 91.143 | 99.143 | 93.143 | 0.940 | 332 | 90.964 | 0.941 | 86.857 |
| opus-med | sonnet | 350 | 97.429 | 0.960 | 226 | 96.018 | 0.944 | 89.429 | 99.714 | 92.000 | 0.940 | 332 | 89.157 | 0.942 | 86.286 |

## Restricted to items where all labellers agree on function

Overall

| n_items | n_labelers | function_fleiss_kappa | function_all_agree_pct | topic_n_items | topic_all_agree_pct | topic_fleiss_kappa | direction_all_exact_pct | direction_alpha_interval | direction_subst_alpha_interval | full_match_all_pct | n_items_labelled_by_any |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 336 | 3 | 1.000 | 100.000 | 218 | 96.330 | 0.966 | 88.988 | 0.960 | 0.960 | 87.202 | 350 |

Pairwise

| labeler_a | labeler_b | n_items | function_pct | function_kappa | topic_n_items | topic_pct | topic_kappa | direction_exact_pct | direction_within1_pct | direction_sign_pct | direction_qwk | direction_subst_n_items | direction_subst_exact_pct | direction_subst_qwk | full_match_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus-low | opus-med | 336 | 100.000 | 1.000 | 218 | 99.541 | 0.994 | 95.238 | 100.000 | 95.833 | 0.975 | 320 | 95.000 | 0.975 | 94.940 |
| opus-low | sonnet | 336 | 100.000 | 1.000 | 218 | 96.789 | 0.955 | 91.964 | 99.702 | 94.048 | 0.953 | 320 | 91.562 | 0.953 | 90.476 |
| opus-med | sonnet | 336 | 100.000 | 1.000 | 218 | 96.330 | 0.948 | 90.476 | 100.000 | 92.560 | 0.950 | 320 | 90.000 | 0.950 | 88.690 |

Items with any disagreement on the primary label: 57
