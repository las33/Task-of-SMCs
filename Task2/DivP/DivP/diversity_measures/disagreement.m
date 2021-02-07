% Disagreement [Skalak 1996, Ho, 1998]
function [value] = disagreement(Hi, Hj, LABELS)
  % para problemas multiclasse, o dividendo é:
  % (a soma da matriz - a diagonal principal) / soma da matriz

  [a, b, c, d] = build_relationship_table(Hi, Hj);
  value = (b + c)/(a + b + c + d);
end
