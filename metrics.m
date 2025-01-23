function [annRet, annVol, Sharpe, MaxDD, Calmar] = metrics(ret, w_ptf)
    % This function computes key performance metrics for a portfolio.

    % Inputs:
    %   - ret: Matrix of asset returns
    %   - w_ptf: Vector of portfolio weights

    % Outputs:
    %   - annRet: Annualized return of the portfolio.
    %   - annVol: Annualized volatility (standard deviation) of portfolio returns.
    %   - Sharpe: Sharpe ratio of the portfolio 
    %   - MaxDD: Maximum drawdown of the portfolio
    %   - Calmar: Calmar ratio 

    % Compute the portfolio's cumulative equity curve
    equity = cumprod(ret * w_ptf);
    
    % Normalize the equity curve to start at 100
    equity = 100 .* equity / equity(1);

    % Calculate performance metrics using a helper function
    [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(equity);
    
end