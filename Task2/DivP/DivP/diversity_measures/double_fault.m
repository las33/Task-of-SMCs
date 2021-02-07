% double_fault: implements giacinto and roli double fault measure
function [diversity] = double_fault(Hi, Hj, LABELS)
  % para mais de duas classes, m vai ser a soma de todos os elementos da matriz

  [a, b, c, d] = build_relationship_table(Hi, Hj);
  m = a + b + c + d;
  e = b + c;

  diversity = e / m;
end
