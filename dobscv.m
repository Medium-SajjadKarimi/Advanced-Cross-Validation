% This function is the extended and modified version fo below link to be consistent with MATLAB `cvpartition`
% Jan Motl (2025). Distribution-balanced stratified cross-validation (https://www.mathworks.com/matlabcentral/fileexchange/72963-distribution-balanced-stratified-cross-validation), MATLAB Central File Exchange. Retrieved February 13, 2025.
% Reference:
%   Study on the impact of partition-induced dataset shift on k-fold
%   cross-validation by Zeng, Xinchuan & Martinez, Tony R.


function cv = dobscv(x, y, n)
%DOBSCV - Distribution-Balanced Stratified Cross-Validation (DB-SCV)
%
% This function partitions data into `n` folds, ensuring that each fold:
% 1. Contains approximately equal samples from each class.
% 2. Assigns nearest neighbors to the same fold to reduce sample dependency.
%
% OUTPUT FORMAT:
% - Similar to `cvpartition`, returns a struct array `cv` with fields:
%   - `cv(f).train_idx`: Logical vector of training indices for fold `f`.
%   - `cv(f).test_idx`: Logical vector of test indices for fold `f`.
%
% INPUTS:
% - `x` (NxD matrix): Feature matrix (N samples, D features).
% - `y` (Nx1 vector): Class labels (numeric, char, or string).
% - `n` (integer): Number of folds.
%
% OUTPUT:
% - `cv` (1×n struct array): Cross-validation partitions.

% Validate inputs
validateattributes(x, {'numeric'}, {'2d'}) % Must be a 2D numeric matrix
validateattributes(y, {'numeric', 'char', 'string'}, {'column'}) % Column vector
validateattributes(n, {'numeric'}, {'scalar', 'positive', 'integer'}) % Must be a positive integer
assert(size(x, 1) == size(y, 1), 'Row count in x and y must match')

% Initialize variables
numSamples = size(y, 1);
solution = nan(numSamples, 1); % Stores fold assignments
classes = unique(y)'; % Get unique class labels
fold = 1; % Tracks the next fold assignment

for class = classes
    % Extract indices of the current class
    i = find(y == class);

    % Randomly shuffle indices to prevent ordering bias
    i = i(randperm(length(i)));
    xT = x'; % Transpose `x` for better cache locality in distance computation

    while ~isempty(i)
        % Compute distances to the first sample in the list
        distances = sum((xT(:, i) - xT(:, i(1))).^2);
        [~, i2] = sort(distances); % Sort indices by distance

        % Select `nrow` nearest neighbors to assign to folds
        nrow = min(n, length(i));
        i2 = i2(1:nrow);

        % Assign these samples to folds cyclically
        solution(i(i2)) = mod(fold:fold+nrow-1, n);
        fold = fold + nrow; % Move to the next fold

        % Remove assigned samples
        i(i2) = [];
    end
end
% Convert solution to `cvpartition`-like structure
cv = struct([]);
for f = 1:n
    cv(f).train_idx = solution ~= f-1; % Training indices (logical)
    cv(f).test_idx = solution == f-1;  % Testing indices (logical)
    cv(f).NumTestSets = n;
end



end
