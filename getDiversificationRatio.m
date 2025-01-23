function DR = getDiversificationRatio(x, Ret)
    % Calculate the diversification ratio of a portfolio.

    % INPUTS:
    %   x   - Column vector of portfolio weights.
    %   Ret - Matrix of asset returns.

    % OUTPUT:
    %   DR  - Diversification ratio

    vola = std(Ret);
    V = cov(Ret);
    
    % Calculate the portfolio volatility
    volaPtf = sqrt(x' * V * x);
    
    % Compute the diversification ratio 
    DR = (x' * vola') / volaPtf;
end
