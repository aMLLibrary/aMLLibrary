def independentTest








function P = indepTest(X,Y,conf)
	
	% Replace 1 if independent
	% Replace 0 if dependent

	if size(X,1) ~= size(Y,1)
		error('The two input must have the same number of rows, i.e. the same number of samples.')
	end
	
	if conf < 0.785
		error('The confidence level for the test cannot be less than 0.785.')
	end
	
	N = size(X,1);
	
	XX = repmat(X,[1 1 N]);
	XXp = permute(XX,[3 2 1]);
	DIFF_X = XX-XXp;
	NORM_X = squeeze(sqrt(sum(DIFF_X.^2,2)));
	X_MATRIX = NORM_X -repmat(mean(NORM_X,1),N,1) -repmat(mean(NORM_X,2),1,N) + mean(mean(NORM_X));
	
	YY = repmat(Y,[1 1 N]);
	YYp = permute(YY,[3 2 1]);
	DIFF_Y = YY-YYp;
	NORM_Y = squeeze(sqrt(sum(DIFF_Y.^2,2)));
	Y_MATRIX = NORM_Y -repmat(mean(NORM_Y,1),N,1) -repmat(mean(NORM_Y,2),1,N) + mean(mean(NORM_Y));
	
	Vxy = mean(mean(X_MATRIX.*Y_MATRIX));
	
	alpha = 1-conf;
	T = N*Vxy/(mean(mean(NORM_X))*mean(mean(NORM_Y)));
	P = T <= norminv(1-alpha/2)^2;
% 	P = T;

end
