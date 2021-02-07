
function [DATASET_HARD,DATASET_HARD_LABELS,DATASET_EASY,DATASET_EASY_LABELS] = kdn(DATA,LABELS, INSTANCES, K)
   
   index = knnsearch(DATA, INSTANCES, "K", K+1);   
   %index(:,1) = [];
   
   output = zeros(length(INSTANCES), 1);
   
   DATASET_HARD = [];
   DATASET_HARD_LABELS = [];
   DATASET_EASY = [];
   DATASET_EASY_LABELS = [];
   
   for i = 1:length(INSTANCES)
      label = LABELS(i);
      sum_k = 0;
      for j = 2:K+1
          if label ~= LABELS(index(i,j))              
            sum_k = sum_k + 1;
          end
      end
      output(i) = sum_k/K; 
   end
  
   for i = 1:length(output)
       if output(i) > 0.5
           DATASET_HARD = [DATASET_HARD; DATA(i,:)]; 
           DATASET_HARD_LABELS = [DATASET_HARD_LABELS; LABELS(i,:)]; 
       else
           DATASET_EASY = [DATASET_EASY;DATA(i,:)];
           DATASET_EASY_LABELS = [DATASET_EASY_LABELS; LABELS(i,:)]; 
       end
   end   
 
end
