function [c, ceq] = nonlinConstr(x, factorLoading, systematicRisk, D, vol_target)
    % nonlinConstr: Defines a nonlinear constraint for portfolio optimization
    
    % Inputs:
    %   x               - Vector of portfolio weights
    %   factorLoading   - Matrix of factor loadings (links factors to assets)
    %   systematicRisk  - Covariance matrix of systematic risk factors
    %   D               - Diagonal matrix of idiosyncratic variances (unsystematic risk)
    %   vol_target      - Target volatility level for the portfolio
    
    % Outputs:
    %   c               - Inequality constraint value (portfolio volatility - target volatility)
    %   ceq             - Equality constraint value (empty in this case)

    % Compute total portfolio volatility
    portfolioVolatility = sqrt((factorLoading' * x)' * systematicRisk * (factorLoading' * x) + x' * D * x);

    % Inequality constraint
    c = portfolioVolatility - vol_target;

    % No equality constraints are defined in this case
    ceq = [];
end
