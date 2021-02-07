function [averaged_q] = averaged_q_statistic(Aq_matrix)
   
  sum_d = 0;
  L = length(Aq_matrix);
  
  for i = 1:L-1
      for j = i:L
          if i == j
              continue;
          end
          sum_d = sum_d + Aq_matrix(i,j);
      end
  end

  averaged_q = (2/(L*(L-1)))*sum_d;
end
