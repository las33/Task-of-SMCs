function [ DECISION_PROFILE ] = build_decision_profile( ensemble, DATA, LABELS )

  DECISION_PROFILE = zeros(length(ensemble), size(DATA, 1));
  
  for ensemble_index = 1:length(ensemble)
    classifier = ensemble{ensemble_index};

    predicted = test(classifier, DATA, LABELS);

    DECISION_PROFILE(ensemble_index, :) = (predicted == LABELS)';
  end
end

