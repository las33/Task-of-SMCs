% Yule 1900
function [value] = q_statistic(Hi, Hj, LABELS)
  [a, b, c, d] = build_relationship_table(Hi, Hj);
  upper = a * d - b * c;
  lower = a * d + b * c;

  value = upper/lower;
end
