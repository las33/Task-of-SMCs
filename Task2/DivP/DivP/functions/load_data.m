function [ DATA, LABELS ] = load_data( filename )
  FILE = importdata(filename);
  
  if isstruct(FILE)
      DATA = FILE.data;
      LABELS = cell2mat(FILE.textdata);
  else
      DATA = FILE(:, 2:size(FILE, 2));
      LABELS = FILE(:, 1);
  end
end

