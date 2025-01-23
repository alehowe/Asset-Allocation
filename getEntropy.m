function e = getEntropy(x)
    % getEntropy - Calculates the entropy of a probability distribution.

    % Input:
    %   x - A column vector representing the probability distribution. 
    %       Each element of x should be non-negative and sum to 1.
    
    % Output:
    %   e - The entropy value, calculated as -sum(x .* log(x)).
    
    e = -x' * log(x);
end
