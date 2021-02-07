% classify_dataset: classify by the dataset
function [ error_rate ] = classify_dataset(ensemble, DATA, LABELS)

  predicted = ensemble_predict(ensemble, DATA, LABELS);

  error_rate = sum(predicted ~= LABELS) / size(DATA, 1);
end
