%%%%%%%%%%%%%%%%%%%%%
% Function: run
%
% Objective: The main function of the system
% 
% Input:
%
%   dataset - The dataset name
%   rebuild - 'true' - if you want to generate new sets to
%                           validation, training and test. 
%             'false' - otherwise
%
% Examples:
%
% >> run('wine',true);
% >> run('wine'); or run('wine',false);
%%%%%%%%%%%%%%%%%%%%%
function [] = run(dataset, rebuild)

  cd DivP
  
  if nargin == 1
    rebuild = false;
  end

  prwarning(0);
  prwaitbar('off');
  warning('off','all');
  graph_destroy;

  folds = 10;
  
  matriz =  []; 

  SIZES = [100];  
  
  
  for L=SIZES
    if rebuild
      build_folds(dataset, L);
    end

    
    for i=1:folds
      load(sprintf('data/%s/%d/fold_%d/validation_1.mat', dataset, L, i));
      load(sprintf('data/%s/%d/fold_%d/validation_2.mat', dataset, L, i)); 
      load(sprintf('data/%s/%d/fold_%d/test.mat',dataset, L, i));
      load(sprintf('data/%s/%d/fold_%d/ensemble.mat',dataset, L, i));
       
      T_DP = build_decision_profile(ensemble, TEST, TEST_LABELS);
      Aq = diversity_graph(ensemble, 'q_statistic', T_DP, TEST_LABELS);
      
      q_init = averaged_q_statistic(Aq); 
      size_init = size(ensemble, 1);
      [acc_init, f1_init, g_mean_init, auc_init] = metrics(ensemble, TEST, TEST_LABELS);
       
      [ ~, ~, ~, ~, ~, ~, ~, ensemble_select] = perform_fold(L, dataset, i, VALIDATION_1, VALIDATION_1_LABELS, VALIDATION_2, VALIDATION_2_LABELS);
    
           
      T_DP = build_decision_profile(ensemble_select, TEST, TEST_LABELS);
      Aq = diversity_graph(ensemble_select, 'q_statistic', T_DP, TEST_LABELS);
      
      q_v = averaged_q_statistic(Aq); 
      size_v = size(ensemble_select, 1);
      [acc_v, f1_v, g_mean_v, auc_v] = metrics(ensemble_select, TEST, TEST_LABELS);
      
      [VALIDATION_HARD_1,VALIDATION_HARD_1_LABELS,VALIDATIONT_EASY_1,VALIDATION_EASY_1_LABELS] = kdn(VALIDATION_1,VALIDATION_1_LABELS,VALIDATION_1, 5);
      [VALIDATION_HARD_2,VALIDATION_HARD_2_LABELS,VALIDATIONT_EASY_2,VALIDATION_EASY_2_LABELS] = kdn(VALIDATION_2,VALIDATION_2_LABELS,VALIDATION_2, 5);
      
      [ ~, ~, ~, ~, ~, ~, ~, ensemble_hard] = perform_fold(L, dataset, i, VALIDATION_HARD_1, VALIDATION_HARD_1_LABELS, VALIDATION_HARD_2, VALIDATION_HARD_2_LABELS);
    
      disp('RESULTS HARD')
      
      T_DP = build_decision_profile(ensemble_hard, TEST, TEST_LABELS);
      Aq = diversity_graph(ensemble_hard, 'q_statistic', T_DP, TEST_LABELS);
      
      q_hard = averaged_q_statistic(Aq); 
      size_hard = size(ensemble_hard, 1);
      [acc_hard, f1_hard, g_mean_hard, auc_hard] = metrics(ensemble_hard, TEST, TEST_LABELS);
      
      [ ~, ~, ~, ~, ~, ~, ~, ensemble_easy] = perform_fold(L, dataset, i, VALIDATIONT_EASY_1, VALIDATION_EASY_1_LABELS, VALIDATIONT_EASY_2, VALIDATION_EASY_2_LABELS);
    
      disp('RESULTS EASY')
      
      T_DP = build_decision_profile(ensemble_easy, TEST, TEST_LABELS);
      Aq = diversity_graph(ensemble_easy, 'q_statistic', T_DP, TEST_LABELS);
      
      q_easy = averaged_q_statistic(Aq); 
      size_easy = size(ensemble_easy, 1);
      [acc_easy, f1_easy, g_mean_easy, auc_easy] = metrics(ensemble_easy, TEST, TEST_LABELS);     

      
      results = [i, q_init, q_v, q_hard, q_easy, acc_init, acc_v, acc_hard, acc_easy, ...
                auc_init, auc_v, auc_hard, auc_easy, g_mean_init, g_mean_v, g_mean_hard, g_mean_easy, ...
                f1_init, f1_v, f1_hard, f1_easy, size_init, size_v, size_hard, size_easy ];
            
      disp('FOLD RESULTS')  
      disp(results)
      matriz = [matriz; results]; 
      writematrix(matriz,sprintf('results_%d.txt',i),'Delimiter',',')
      
    end
    
    writematrix(matriz,'results.txt','Delimiter',',')
    %%%%%%%%%
    %
    % Code to save the results on disc. Please create the folder 'results'
    %
    %%%%%%%%%
    %filenameMAT = strcat('matriz/', dataset,  num2str(L) , '.mat');
    %save(filenameMAT);
  end
  
  cd ..
  
end

function [architecture_error, architecture_weights, architecture_size, architecture_final_ensemble, ...
  full_ensemble_error, single_best_error, oracle_error, ensemble_select] = perform_fold(L, dataset, i, VALIDATION_1, VALIDATION_1_LABELS, VALIDATION_2, VALIDATION_2_LABELS)
  fprintf('Run %d #%d\n', L, i);

  load(sprintf('data/%s/%d/fold_%d/ensemble.mat',     dataset, L, i));
  load(sprintf('data/%s/%d/fold_%d/test.mat',         dataset, L, i));
  
  DP = build_decision_profile(ensemble, VALIDATION_1, VALIDATION_1_LABELS);
  
  % Using DP to calculate the diversity between classifiers
  [Adis, Aq, Ap, Ak, Adf] = build_measures_matrixes(ensemble, DP, VALIDATION_1_LABELS);

  %%%%%%%%%%%%%%% Aq NORMALIZATION TO INTERVAL [0;1] %%%%%%%%%%%%%%
  vec = unique(Aq);
  maxVec = vec(end-1);
  minVec = vec(1);
  Aq = ((Aq-minVec)./(maxVec-minVec));
  Aq(Aq >= 16) = 32;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  Adis = 1 - Adis;
  Adis(Adis <= -31) = 32;
  
  FitnessHolder = @(X)ga_fitness_function(X, ensemble, Adis, Aq, Ap, Ak, Adf, VALIDATION_2, VALIDATION_2_LABELS);

  ip = [1,1,1,1,1,-5];
  ip = [ip;[-1,-1,-1,-1,-1,5]];
  
  graph_init(1500);
  BEST = ga(FitnessHolder, 6, [],[],[],[],[],[],[], ...
    gaoptimset('Display', 'iter', 'Vectorized', 'on', ...
      'PopulationSize', 22, 'PopInitRange', [-1;1], 'FitnessLimit', 0, 'StallGenLimit', 15, ...
      'InitialPopulation',ip,'EliteCount',3));
  graph_destroy;

  job_full = batch(@classify_dataset, 1, {ensemble, TEST, TEST_LABELS}, 'AdditionalPaths', 'prtools');
  job_oracle = batch(@oracle_classify, 1, {ensemble, TEST, TEST_LABELS}, 'AdditionalPaths', 'prtools');
  job_sbe = batch(@single_best_classifier, 1, {ensemble, TEST, TEST_LABELS}, 'AdditionalPaths', 'prtools');
  [best_ensemble, ~] = find_best(ensemble, Adis, Aq, Ap, Ak, Adf, VALIDATION_2, VALIDATION_2_LABELS, BEST(1), BEST(2), BEST(3), BEST(4), BEST(5), BEST(6));

  wait(job_oracle);
  wait(job_sbe);
  wait(job_full);

  architecture_error = test_best(dataset, L, i, ensemble(best_ensemble));
  architecture_weights = BEST;
  architecture_size = length(best_ensemble);
  %architecture_final_ensemble = sprintf('%d-', best_ensemble);
  architecture_final_ensemble = best_ensemble;
  ensemble_select = ensemble(best_ensemble);
  
  full_output = fetchOutputs(job_full);
  full_ensemble_error = full_output{1}; % Erro do ensemble por fold

  sbe_output = fetchOutputs(job_sbe);
  single_best_error = sbe_output{1};

  oracle_output = fetchOutputs(job_oracle);
  oracle_error = oracle_output{1};

  delete(job_oracle);
  delete(job_sbe);
  delete(job_full);
  
end

function [new_ensemble, error_rate] = find_best(ensemble, Adis, Aq, Ap, Ak, Adf, DATA, LABELS, w_dis, w_q, w_p, w_k, w_df, t)
  graph_init;

  Af = Adis * w_dis + Aq * w_q + Ap * w_p + Ak * w_k + Adf * w_df;

  % To eliminate the main diagonal
  MASK = xor(eye(length(ensemble)), ones(length(ensemble)));
  ADJACENCY_MATRIX = Af < t & MASK;

  [new_ensemble, error_rate] = build_color_ensemble(ensemble, ADJACENCY_MATRIX, DATA, LABELS);

  graph_destroy;
end

function [error_rate] = test_best(dataset, L, i, best_ensemble)
  load(sprintf('data/%s/%d/fold_%d/test.mat', dataset, L, i));

  error_rate = classify_dataset(best_ensemble, TEST, TEST_LABELS);  
  
  graph_destroy;
end


