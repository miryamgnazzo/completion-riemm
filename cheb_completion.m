function [Res, Xfinal, max_err, max_relerr] = cheb_completion(core_dims, final_dims, sizes, intervals, pi0, en, fun)
%function for riemannian tensor completion, with two step of chebyshev
%approximation in cheb_riemm

    if ~exist('tenrand', 'file')
        fprintf('Tensor Toolbox version 2.6 or higher is required.\n');
        return;
    end

   d = length(intervals);

   if length(sizes) < 3
        error('missing intermediate size');
   end

   n0 = sizes(1);
   n1 = sizes(2);
   n2 = sizes(3);

    if (max(core_dims) > n0)
        error('check the rank');
    end

   %qui devo costruire questi values!!
    values1 = cell(1,d);
    for i = 1:d
        values1{i} = chebpts(n1, intervals{i});
    end
   
    options.maxiter = 50;
    options.maxinner = 50;
    options.maxtime = 100;
    options.storedepth = 3;
    % Target gradient norm
    options.tolgradnorm = 1e-3;

    coefficient = 0.5; %20; %(questo con kolmog) 
    %0.5; %(questo nel caso exp)

 [Res1, X1] = cheb_riemm(core_dims, n1*ones(1,d), n0, values1, intervals, pi0, en, options, coefficient);
 %sparse sotto
 %[Res1, X1] = cheb_riemm_sparse(core_dims, n1*ones(1,d), n0, values1, intervals, pi0, en, options, coefficient);


  %check the error
  [vals1, valsXt1, err1, relerr1] = check_samples(X1, values1, 5000, pi0, en);
   %[vals1, valsXt1, err1, relerr1] = check_samples_sparse(X1, values1, 5000, pi0, en);


    values2 = cell(1,d);
    for i = 1:d
        values2{i} = chebpts(n2, intervals{i});
    end

    options.maxiter = 100;
    options.maxinner = 50;
    %options.maxtime = inf;
    %options.storedepth = 3;
    % Target gradient norm
    options.tolgradnorm = 1e-4;

    coefficient = 2;  %50; %(questo con kolmog) 
    %2; %(questo nel caso exp)

   [Res2, X2] = cheb_riemm(core_dims, n2*ones(1,d), n1, values2, intervals, pi0, en, options, coefficient, X1);
   %[Res2, X2] = cheb_riemm_sparse(core_dims, n2*ones(1,d), n1, values2, intervals, pi0, en, options, coefficient, X1);


   %check the error
   [vals2, valsXt2, err2, relerr2] = check_samples(X2, values2, 5000, pi0, en);
   %[vals2, valsXt2, err2, relerr2] = check_samples_sparse(X2, values2, 5000, pi0, en);


    values3 = cell(1,d);
    for i = 1:d
        values3{i} = chebpts(final_dims(i), intervals{i});
    end

    options.maxiter = 100;
    options.maxinner = 100;
    %options.maxtime = inf;
    %options.storedepth = 3;
    % Target gradient norm
    options.tolgradnorm = 1e-5;

    coefficient = 5;  %100;  %(questo con kolmog) 
  %5; %(questo nel caso exp)

  [Res3, Xfinal] = cheb_riemm(core_dims, final_dims, n2, values3, intervals, pi0, en, options, coefficient, X2);
   %[Res3, Xfinal] = cheb_riemm_sparse(core_dims, final_dims, n2, values3, intervals, pi0, en, options, coefficient, X2);
  
  %check the error
  [vals3, valsXt3, err3, relerr3] = check_samples(Xfinal, values3, 5000, pi0, en);
  %[vals3, valsXt3, err3, relerr3] = check_samples_sparse(Xfinal, values3, 5000, pi0, en);

  %Final outputs
   Res = [Res1, Res2, Res3];
   max_err = [max(err1), max(err2), max(err3)];
   max_relerr = [max(relerr1), max(relerr2), max(relerr3)];

end