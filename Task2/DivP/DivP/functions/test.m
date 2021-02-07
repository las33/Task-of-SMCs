function [ classes ] = test( classifier, examples, labels )
  classes = test_perceptron(classifier, examples, labels);
end

function [classes] = test_perceptron(perceptron, examples, labels)
  classes = labeld(prdataset(examples, labels), perceptron);
end

function [class] = test_nn(nn, example)
  Y = nn(example');
  class = round(Y);
end

function [class] = test_knn(knn, example)
  class = predict(knn, example);
end
