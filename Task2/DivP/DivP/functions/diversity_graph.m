function [ DIVERSITY_TABLE ] = diversity_graph( ensemble, function_name, DP, LABELS )
  DIVERSITY_TABLE = repmat(32, length(ensemble), length(ensemble)); % 32 to avoid this creating a matrix with a valid correlation

  for i = 1:length(ensemble)
    for j = i:length(ensemble)
      if i == j
        continue;
      end

      DIVERSITY_TABLE(i, j) = feval(function_name, DP(i, :), DP(j, :), LABELS);
      DIVERSITY_TABLE(j, i) = DIVERSITY_TABLE(i, j);
    end
  end
end
