function [a, b, c, d] = build_relationship_table(CLASSIFIER_1_DATA, CLASSIFIER_2_DATA)
  step = 1 / length(CLASSIFIER_1_DATA);

  a = step/length(CLASSIFIER_1_DATA);
  b = step/length(CLASSIFIER_1_DATA);
  c = step/length(CLASSIFIER_1_DATA);
  d = step/length(CLASSIFIER_1_DATA);

  for i = 1:length(CLASSIFIER_1_DATA)
    if CLASSIFIER_1_DATA(i) == CLASSIFIER_2_DATA(i)
      if CLASSIFIER_1_DATA(i) == 1
        a = a + step;
      else
        d = d + step;
      end
    elseif CLASSIFIER_1_DATA(i) ~= CLASSIFIER_2_DATA(i)
      if CLASSIFIER_1_DATA(i) == 1
        b = b + step;
      else
        c = c + step;
      end
    end
  end
end