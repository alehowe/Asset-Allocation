function vols = getVolContributions(x, LogRet)
    % This function calculates the volatility contributions of assets 
    % based on their weights and historical log returns.

    % Inputs:
    %   x - Vector of asset weights (e.g., portfolio allocations).
    %   LogRet - Matrix of historical logarithmic returns of the assets

    % Output:
    %   vols - Vector containing the contribution of each asset

    % Calculate the weighted variances of each asset.
    weighted_variances = (x.^2) .* (std(LogRet)'.^2);

    % Normalize the weighted variances to calculate contributions.
    vols = weighted_variances / sum(weighted_variances);
end
