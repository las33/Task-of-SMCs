%%%%%%%%%%%%%%%%%%%%%
% Function: train_ensemble_folds 
% 
% Objective: Create the ensemble of classifiers
%
% Input:
%
%   ensemble_size - The ensemble size
%   TRAIN_DATA - The data to train a classifier
%   TRAIN_LABELS - The data labels to train a classifier
%
% Output:
%
%   ensemble - A cell struct with all the classifiers
%
%%%%%%%%%%%%%%%%%%%%%
function [ensemble] = train_ensemble_folds( ensemble_size, TRAIN_DATA, TRAIN_LABELS )
    ensemble = cell(ensemble_size,1);

    for i = 1:ensemble_size
      [BAGGED_DATA, BAGGED_LABELS] = bagging(TRAIN_DATA, TRAIN_LABELS);
      
      ensemble{i} = train(BAGGED_DATA, BAGGED_LABELS);
    end
end

