function CumExplainedVar = getCumulativeExplainedVar(latent, n)
    % This function calculates the cumulative explained variance for the 
    % first n principal components based on their eigenvalues.

    % Input:
    %   latent - A vector containing the eigenvalues of the covariance matrix.
    %            These represent the variance explained by each principal component.
    %   n - The number of principal components to consider for the cumulative sum.

    % Output:
    %   CumExplainedVar - The cumulative proportion of variance explained
    %                     by the first n principal components.

    % Compute the proportion of variance explained by the first n components
    ExplainedVar = latent(1:n) / sum(latent);
    
    % Compute the cumulative explained variance by summing the proportions
    CumExplainedVar = sum(ExplainedVar);
end
