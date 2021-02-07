%%%%%%%%%%%%%%%%%%%%%
% Function: train 
% 
% Objective: Train a classifier
%
% Input:
%
%   DATA - The data to train a classifier
%   LABELS - The data labels to train a classifier
%
% Output:
%
%   classifier - The classifier trained
%
%%%%%%%%%%%%%%%%%%%%%
function [ classifier ] = train( DATA, LABELS )

    %%%
    % You can change here to train another classifier as a KNN or DECISION
    % TREE, for example.
    %%%
    classifier = train_perceptron(DATA, LABELS);

end

function [perceptron] = train_perceptron(DATA, LABELS)
  A = prdataset(DATA, LABELS);

  perceptron = perlc(A, 50);
end

function [knn] = train_knn(DATA, LABELS)
  knn = ClassificationKNN.fit(DATA, LABELS, 'NumNeighbors', 1);
end