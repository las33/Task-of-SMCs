% oracle_classify: classify by the ensembles oracle
function [ oracle_error_rate ] = oracle_classify(ensemble, TEST, TEST_LABELS)

  correct = 0;

  for i=1:size(TEST, 1)
    for k=1:length(ensemble)
      class = test(ensemble{k}, TEST(i, :), TEST_LABELS(i));
      if class == TEST_LABELS(i)
        correct = correct + 1;
        break;
      end
    end
  end

  oracle_error_rate = (size(TEST_LABELS, 1) - correct) / size(TEST_LABELS, 1);
end
