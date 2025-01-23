function s = getSharpeRatio(x, ExpRet, V, rf)
    % This function computes the Sharpe Ratio
    % Inputs:
    %   x      - Vector of portfolio weights
    %   ExpRet - Vector of expected returns for the assets
    %   V      - Covariance matrix of asset returns
    %   rf     - Risk-free rate
    %
    % Output:
    %   s      - Sharpe Ratio of the portfolio

    % Portfolio excess return over the risk-free rate
    excessReturn = x' * ExpRet' - rf;
    
    % Portfolio risk (standard deviation of returns)
    portfolioRisk = sqrt(x' * V * x);
    
    % Sharpe Ratio: Measure of risk-adjusted return
    s = excessReturn / portfolioRisk;
end
