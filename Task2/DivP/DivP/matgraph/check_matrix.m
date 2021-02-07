function ok = check_matrix(A)
% check_matrix(A) --- check if A is a valid adjacency matrix for a graph

ok = 0;

% make sure matrix is equare
[nr,nc] = size(A);
if (nr ~= nc)
  fprintf('matrix not square?');
    return
end

% see if the matrix is zero-one
B = (A>0); % convert to logical (so 0,1).
if (nnz(A-B)>0)
  fprintf('matrix not zero-one?');
    return
end

% check if symmetric
if (nnz(A-A')>0)
  fprintf('matrix not symmetric?');
    return
end

% check the diagonal
if (any(diag(A)>0))
  fprintf('matrix not diagonal????');
    return
end

% all tests passed
ok = 1;
