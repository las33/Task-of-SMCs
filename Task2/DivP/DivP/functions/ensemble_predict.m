function [ predicted_classes ] = ensemble_predict( ensemble, DATA, LABELS )

    MAX_CLASS_COUNT = 40;
    DISTINCT_LABELS = unique(LABELS);

    class_scores = zeros(size(LABELS, 1), MAX_CLASS_COUNT);

    for i = 1:length(ensemble)
      classes = test(ensemble{i}, DATA, LABELS);

      for j=1:length(DISTINCT_LABELS)
        index = DISTINCT_LABELS(j);

        if index <= length(DISTINCT_LABELS)
          class_scores(:, index) = (classes == index) + class_scores(:, index);
        end
      end
    end

    [~, predicted_classes] = max(class_scores, [], 2);
end

