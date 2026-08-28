function run_ex3_all_ne(cfg)
%RUN_EX3_ALL_NE  Driver for Case study 2 reliability: ne = 20, 30, 40.
%   Each value of ne writes its own .mat and runs with resume enabled, so
%   re-launching the driver continues an interrupted run.
%
%   run_ex3_all_ne
%   run_ex3_all_ne(struct('ne_list',[20 30]))

if nargin < 1, cfg = struct(); end
if isfield(cfg,'ne_list'), ne_list = cfg.ne_list; else, ne_list = [20 30 40]; end
cfg = rmfieldif(cfg, 'ne_list');

here = fileparts(mfilename('fullpath'));
master = fullfile(here, 'run_ex3_all_ne_progress.txt');
fid = fopen(master,'a');
fprintf(fid, '\n===== driver avviato %s | ne_list = [%s] =====\n', ...
    datestr(now), num2str(ne_list));
fclose(fid);

t0 = tic;
esiti = strings(1, numel(ne_list));
for i = 1:numel(ne_list)
    ne = ne_list(i);
    c = cfg; c.ne = ne;
    if ~isfield(c,'resume'), c.resume = true; end   % sempre riprendibile
    logline(master, sprintf('--- avvio ne = %d (%d/%d) ---', ne, i, numel(ne_list)));
    tne = tic;
    try
        Script_case2_compare_reliability_seed50(c);
        esiti(i) = sprintf('ne=%d OK (%.2f h)', ne, toc(tne)/3600);
    catch ME
        esiti(i) = sprintf('ne=%d FALLITO: %s', ne, ME.message);
        logline(master, sprintf('ERRORE su ne = %d: %s', ne, getReport(ME,'basic')));
    end
    logline(master, sprintf('--- fine ne = %d: %s ---', ne, esiti(i)));
end

logline(master, sprintf('===== driver concluso in %.2f h =====', toc(t0)/3600));
for i = 1:numel(esiti), logline(master, sprintf('   %s', esiti(i))); end
end

function logline(f, msg)
    fid = fopen(f,'a'); fprintf(fid, '%s  %s\n', datestr(now,'HH:MM:SS'), msg); fclose(fid);
    fprintf('%s\n', msg);
end

function c = rmfieldif(c, f)
    if isfield(c,f), c = rmfield(c,f); end
end
