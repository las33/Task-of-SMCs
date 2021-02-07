% Kappa-Statistic [Cohem 1960]; [Dietterich, 2000b]
function [value] = kappa_statistic(Hi, Hj, LABELS)
  [a, b, c, d] = build_relationship_table(Hi, Hj);
  m = a + b + c + d;
  o1 = (a + d) / m; % soma da matriz principal
  o2 = ((a + b)*(a + c) + (c + d)*(b + d)) /(m * m);

  value = (o1 - o2) / (1 - o2);
end
