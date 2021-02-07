% classify_dataset: classify by the dataset
function [ accuracy, f_measure, gmean, AUC] = metrics(ensemble, DATA, LABELS)

  predicted = ensemble_predict(ensemble, DATA, LABELS);   
  
 retorno = Evaluate(LABELS,predicted) ;
 accuracy = retorno(1);
 f_measure = retorno(6);
 gmean = retorno(7);
 [~,~,~, AUC] = perfcurve(LABELS,predicted,1);
 
  
end
