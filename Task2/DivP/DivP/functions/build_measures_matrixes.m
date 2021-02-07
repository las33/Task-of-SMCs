function [Adis, Aq, Ap, Ak, Adf] = build_measures_matrixes(ensemble, DP, VALIDATION_1_LABELS)
  Adis = diversity_graph(ensemble, 'disagreement', DP, VALIDATION_1_LABELS);
  Aq = diversity_graph(ensemble, 'q_statistic', DP, VALIDATION_1_LABELS);
  Ap = diversity_graph(ensemble, 'correlation', DP, VALIDATION_1_LABELS);
  Ak = diversity_graph(ensemble, 'kappa_statistic', DP, VALIDATION_1_LABELS);
  Adf = diversity_graph(ensemble, 'double_fault', DP, VALIDATION_1_LABELS);
end
