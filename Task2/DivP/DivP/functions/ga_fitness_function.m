% ga_fitness_function: function description
function [fitness] = ga_fitness_function(weights, ensemble, Adis, Aq, Ap, Ak, Adf, DATA, LABELS)
  pop_size = size(weights, 1);

  fitness = zeros(pop_size, 1);
  MASK = xor(eye(length(ensemble)), ones(length(ensemble)));
  
  for p = 1:pop_size
    Af = Adis * weights(p, 1) + Aq * weights(p, 2) + Ap * weights(p, 3) + Ak * weights(p, 4) + Adf * weights(p, 5);
    ADJACENCY_MATRIX = Af < weights(p, 6) & MASK;

    [~, fitness_value] = build_color_ensemble(ensemble, ADJACENCY_MATRIX, DATA, LABELS);
    fitness(p) = fitness_value;
  end
  
end
