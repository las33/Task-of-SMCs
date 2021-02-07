% single_best_classifier: returns the error and the index of the best classifier
function [ best_error, index ] = single_best_classifier(ensemble, TEST, TEST_LABELS)
  best_error = 1;
  index = 0;

  for i=1:length(ensemble)
    classifier_error = classify_dataset({ensemble{i}}, TEST, TEST_LABELS);

    if classifier_error < best_error
      best_error = classifier_error;
      index = i;
    end
  end
end
