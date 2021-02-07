%%%%%%%%%%%%%%%%%%%%%
% Function: build_color_ensemble 
% 
% Objective: To select the best ensemble
%
% Input:
%
%   ensemble - The ensemble of classifiers    
%   ADJACENCY_MATRIX - Adjacency Matrix
%   DATA - Original DATA
%   LABELS - Original LABELS
%
% Output:
%
%   new_ensemble - The selection result
%   best_error_rate - The new_ensemble error
%
%%%%%%%%%%%%%%%%%%%%%
function [new_ensemble, best_error_rate] = build_color_ensemble(ensemble, ADJACENCY_MATRIX, DATA, LABELS)
  GRAPH = graph(length(ensemble));
  set_matrix(GRAPH, ADJACENCY_MATRIX);

  COLORS = color(GRAPH);
  groups = parts(COLORS);

  best = 0;
  best_error_rate = 1;

  for i = 1:length(groups)
    error_rate = classify_dataset(ensemble(groups{i}), DATA, LABELS);

    if error_rate <= best_error_rate
      best_error_rate = error_rate;
      best = i;
    end
  end
  
  new_ensemble = groups{best};
end