function [annRet, annVol, Sharpe, MaxDD, Calmar] = getPerformanceMetrics(x)
% Calculate key performance metrics for a financial time series.

% INPUT:
%   x - Vector representing the time series of asset prices or portfolio values.

% OUTPUT:
%   annRet  - Annualized return
%   annVol  - Annualized volatility of the returns.
%   Sharpe  - Sharpe ratio
%   MaxDD   - Maximum drawdown
%   Calmar  - Calmar ratio

% Annualized return
annRet = (x(end)/x(1))^(1/(length(x)/252)) - 1;

% Annualized volatility
annVol = std(tick2ret(x)) * sqrt(252);

% Sharpe ratio
Sharpe = annRet / annVol;

Draw_down = zeros(1, length(x));
for i = 1:length(x)
    Draw_down(i) = (x(i) / max(x(1:i))) - 1;
end
MaxDD = min(Draw_down); % Maximum drawdown

% Calmar ratio
Calmar = annRet / abs(MaxDD);

end