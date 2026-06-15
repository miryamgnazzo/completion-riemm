function A = eval_all(values, pi0, en, fun)
% values = {tvec, theta1, theta2, ...}
% pi0, en, fun già definiti

if length(values) < 2
    error('missing parameters');
end

tensor_dims = cellfun(@length, values);
d = length(tensor_dims);

A = zeros(tensor_dims);
tvec = values{1};

% loop su tutte le combinazioni dei parametri
subs = cell(1, d-1);
all_ind = 1:prod(tensor_dims(2:end));
[subs{:}] = ind2sub(tensor_dims(2:end), all_ind);

for k = 1:length(all_ind)

    param = cell(1, d-1);
    idx = cell(1, d);
    idx{1} = 1:tensor_dims(1);  % fibra temporale

    for j = 2:d
        idx{j} = subs{j-1}(k);
        param{j-1} = values{j}(idx{j});
    end

    v = fiber_time(tvec, param, pi0, en); %11-06
%    v = fiber_time_fast(tvec, param, pi0, en); %

    A(idx{:}) = v;
end

%Debug
%[vals, valsXt, err, relerr] = check_samples(A, values, 500, pi0, en);
%keyboard