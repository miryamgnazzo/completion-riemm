function setup_paths()
%SETUP_PATHS  Add the shared functions of this repository to the MATLAB path.
%   Called automatically by every experiment script.
%
%   External dependencies, not included here and to be added separately:
%   Manopt, Tensor Toolbox and MarkovCrossApproximation (ACA). See README.md.

root = fileparts(mfilename('fullpath'));

addpath(fullfile(root, 'src', 'completion'));
addpath(fullfile(root, 'src', 'models'));

end
