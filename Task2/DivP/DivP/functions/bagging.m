%%%%%%%%%%%%%%%%%%%%%
% Function: bagging 
% 
% Input:
%
%   ORIGINAL_DATA - original data
%   ORIGINAL_LABELS - original labels
%
% Output:
%
%   DATA - sample data
%   LABELS - sample labels
%
%%%%%%%%%%%%%%%%%%%%%
function [DATA, LABELS] = bagging(ORIGINAL_DATA, ORIGINAL_LABELS)
  data_length = length(ORIGINAL_DATA);
  RAND_INDEXES = randi(data_length, 1, data_length);

  DATA = ORIGINAL_DATA(RAND_INDEXES, :);
  LABELS = ORIGINAL_LABELS(RAND_INDEXES);
end