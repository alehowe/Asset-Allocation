function [relRC, RC, mVol] = getRiskContributions(x, Ret)
    % Calculate the risk contributions of portfolio assets.
    % Inputs:
    %   x   - Column vector of portfolio weights
    %   Ret - A matrix of asset returns
    
    % Outputs:
    %   relRC - Column vector of relative risk contributions of each asset to the portfolio's total risk.
    %   RC    - Column vector of absolute risk contributions of each asset to the portfolio's total risk.
    %   mVol  - Column vector of marginal volatilities of each asset in the portfolio.
    
    V = cov(Ret);
    VolaPtf = sqrt(x' * V * x);
    
    % Compute the marginal contribution to portfolio volatility for each asset.
    mVol = V * x / VolaPtf;
    
    % Compute the absolute risk contribution of each asset.
    RC = mVol .* x;
    
    % Compute the relative risk contribution of each asset as a proportion of total risk.
    relRC = RC / sum(RC);
end