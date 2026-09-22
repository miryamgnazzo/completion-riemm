function run_ex3_underrepair_paper(cfg)
%RUN_EX3_UNDERREPAIR_PAPER  Driver for Case study 2 under repair.
%   Runs ne = 20, 30, 40 in the setting: d = 5, core [16 4 4 4 4],
%   the reliability parameter order and the remaining parameters fixed at
%   the lower bound.
%
%   run_ex3_underrepair_paper
%   run_ex3_underrepair_paper(struct('ne_list',20))

if nargin < 1, cfg = struct(); end
if isfield(cfg,'ne_list'), ne_list = cfg.ne_list; cfg = rmfield(cfg,'ne_list');
else,                      ne_list = [20 30 40];
end

here   = fileparts(mfilename('fullpath'));
master = fullfile(here, 'run_ex3_underrepair_paper_progress.txt');
logline(master, sprintf('===== driver avviato %s | ne_list = [%s] =====', ...
        datestr(now), num2str(ne_list)));

t0 = tic;
for i = 1:numel(ne_list)
    ne = ne_list(i);

    c              = cfg;
    c.ne           = ne;
    c.d            = 5;
    c.ncheb        = [16 16];
    c.coeff_levels = [1 6];
    c.core_list    = { [16 4 4 4 4] };
    c.nsamples     = 2000;
    c.par_order    = [3 4 5 7 8 9 10 11 12 6 1 2];   % ordine di reliability
    c.fix_at       = 'lower';                        % fissi al bordo inferiore
    c.outfile      = sprintf('case2_underrepair_ordREL_LOWER_d5_ne%d_r4.mat', ne);
    c.logfile      = sprintf('case2_underrepair_ordREL_LOWER_d5_ne%d_r4_progress.txt', ne);
    if ~isfield(c,'resume'), c.resume = true; end

    logline(master, sprintf('--- avvio ne = %d (%d/%d) ---', ne, i, numel(ne_list)));
    tne = tic;
    try
        Script_case2_underrepair_reward_paper(c);
        logline(master, sprintf('--- ne = %d OK (%.1f min) ---', ne, toc(tne)/60));
    catch ME
        logline(master, sprintf('--- ne = %d FALLITO: %s ---', ne, ME.message));
        logline(master, getReport(ME,'basic'));
    end
end
logline(master, sprintf('===== driver concluso in %.1f min =====', toc(t0)/60));
end

function logline(f, msg)
    fid = fopen(f,'a'); fprintf(fid, '%s  %s\n', datestr(now,'HH:MM:SS'), msg); fclose(fid);
    fprintf('%s\n', msg);
end
