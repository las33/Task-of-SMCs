function [value] = correlation(Hi, Hj, LABELS)
  [a, b, c, d] = build_relationship_table(Hi, Hj);
  upper = (a * d) - (b * c);
  lower = sqrt((a+b)*(c+d)*(a+c)*(b+d));
  
  value = upper/lower;
end
