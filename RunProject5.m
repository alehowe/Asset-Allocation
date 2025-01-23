clear all
close all
clc

%% Read Prices
% path_map = 'C:\Users\carlo\MATLAB Drive\COMPUTATIONAL FINANCE\';
% path_map = 'C:\Users\matte\MATLAB Drive\COMPUTATIONAL FINANCE\';
% path_map = '/Users/ceciliagaspari/Desktop/COMPUTATIONAL FINANCE/ASSET MANAGMENT/';
% path_map = 'C:\Users\Utente\MATLAB Drive\COMPUTATIONAL FINANCE';

path_map = pwd;
filename = 'prices.xlsx';

%%
table_prices = readtable(fullfile(path_map, filename));

%% Transform prices from table to timetable
dt = table_prices(:,1).Variables;
values = table_prices(:,2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);
myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm); 

%% Selection of a subset of Dates
start_dt = datetime('01/01/2023', 'InputFormat', 'dd/MM/yyyy'); 
end_dt   = datetime('31/12/2023', 'InputFormat', 'dd/MM/yyyy');

rng_dates = timerange(start_dt, end_dt,'closed');
subsample = myPrice_dt(rng_dates,:);

prices_val = subsample.Variables;
dates_ = subsample.Time;

%% Calculate returns
ret = prices_val(2:end,:)./prices_val(1:end-1,:);
LogRet = log(ret);
ExpRet = mean(LogRet);
VVV = cov(ret);
%% Calculate Variance
var_ = var(LogRet);
std_ = std(LogRet);
V = cov(LogRet);

%% Creation of N random portfolio
N = 100000;
NumAssets = size(prices_val,2);
RetPtfs = zeros(1,N);
VolaPtfs = zeros(1,N);
SharpePtfs = zeros(1,N);

for n = 1:N
    w = rand(1,NumAssets);
    w_norm = w./sum(w);
    
    exp_ret_ptf = w_norm*ExpRet';
    exp_vola_ptf = sqrt(w_norm*V*w_norm');
    sharpe_ratio = exp_ret_ptf/exp_vola_ptf;

    RetPtfs(n) = exp_ret_ptf;
    VolaPtfs(n) = exp_vola_ptf;
    SharpePtfs(n) = sharpe_ratio;
end

%% Point 1
% Portfolio Frontier
lb = zeros(NumAssets,1);
ub = ones(NumAssets,1);

p = Portfolio("NumAssets",NumAssets);
p = setDefaultConstraints(p);
p = estimateAssetMoments(p,LogRet,'MissingData',false);
p = p.setBounds(lb, ub);                  
p = p.setBudget(1, 1);                    
pwgt = p.estimateFrontier(100);
[FrontierVola, FrontierRet] = estimatePortMoments(p,pwgt);

%% Plot Frontier
figure()
plot(FrontierVola, FrontierRet);
xlabel('Volatility')
ylabel('Expected return')

%% Plot Frontier and Portfolios
figure();
title('Expected return vs volatility')
scatter(VolaPtfs, RetPtfs, [], SharpePtfs, 'filled')
colorbar
hold on
plot(FrontierVola, FrontierRet)
xlabel('Volatility')
ylabel('Expected return')

%% MAXIMUM SHARPE RATIO and MINIMUM VARIANCE
Port = Portfolio("NumAssets",NumAssets,'Name','MeanVariance');
Port = setDefaultConstraints(Port);
Port = setAssetMoments(Port,ExpRet,V);

[~, w_A_MVP] = Port.estimateFrontierLimits(); % Minimum Variance
w_B_sharp = estimateMaxSharpeRatio(Port); % Max Sharpe Ratio

[VA, RA] = estimatePortMoments(Port, w_A_MVP);
[VB, RB] = estimatePortMoments(Port, w_B_sharp);
%%
figure()
plot(FrontierVola, FrontierRet, 'LineWidth', 2, 'Color', [0 0.5 0.8]); % Efficient frontier in blue-green
hold on

% Plot Portfolio A and B with larger markers and distinct colors
plot(VA(1), RA(1), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio A');
plot(VB, RB, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio B');

% Add legend and axis labels
legend({'Efficient Frontier', 'Portfolio A', 'Portfolio B'}, 'Location', 'best', 'FontSize', 10);
xlabel('Volatility (\sigma)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Expected Return (\mu)', 'FontSize', 12, 'FontWeight', 'bold');

%% Point 2
% Frontier with constraints 
% Initialization
x0 = rand(NumAssets, 1);  % Random initialization of weights
x0 = x0/sum(x0);  % Normalize weights

% Constraints
SensitiveIdx = [1,5,8];
CyclicalIdx = [2,4,6,10,11];
ConsumerStaplesIdx = 7;
LowVolatilityIdx = 16;

% Sensitive and cyclical sector constraints
I = eye(NumAssets);
A_sensitive = -sum(I(SensitiveIdx, :), 1);  % Row vector
b_sensitive = -0.10;
A_cyclical = sum(I(CyclicalIdx, :), 1);    % Row vector
b_cyclical = 0.30;

% Consumer Staples and Low Volatility constraints
A_cs_lv = I([ConsumerStaplesIdx, LowVolatilityIdx], :);
b_cs_lv = [0; 0];

% Maximum exposure constraint
A_max_exposure = zeros(1,NumAssets);
A_max_exposure(1:11) = 1;
b_max_exposure = 0.8;

A = [A_sensitive; A_cyclical; A_max_exposure];
b = [b_sensitive; b_cyclical; b_max_exposure];

% Portfolio optimization
p = Portfolio('NumAssets', NumAssets);
p = setAssetMoments(p, ExpRet, V);
p = setDefaultConstraints(p);
p = p.setInequality(A, b);  % Set custom constraints
p =p.setEquality(A_cs_lv, b_cs_lv);

% Estimate efficient frontier
pwgt = p.estimateFrontier(100);
[FrontierVolaR, FrontierRetR] = estimatePortMoments(p, pwgt);

% Compute Minimum Variance Portfolio (Portfolio C)
[~, w_C_MVP] = p.estimateFrontierLimits();
w_C_MVP = w_C_MVP(:, 1);

% Compute Maximum Sharpe Ratio Portfolio (Portfolio D)
w_D_sharp = p.estimateMaxSharpeRatio();
%%
figure()
plot(FrontierVola, FrontierRet);
xlabel('Volatility')
ylabel('Expected return')
hold on
plot(FrontierVolaR, FrontierRetR);


% %% Check
% A * w_C_MVP <= b
% A * w_D_sharp <= b

%% Plot Frontiers and Max Sharpe Ratio Portfolios
[VC, RC] = estimatePortMoments(p, w_C_MVP);
[VD, RD] = estimatePortMoments(p, w_D_sharp);

%%
figure()
plot(FrontierVolaR, FrontierRetR, 'LineWidth', 2, 'Color', [0.2 0.6 0.3], 'DisplayName', 'Efficient with constraints'); % Efficient in green
hold on
plot(FrontierVola, FrontierRet, 'LineWidth', 2, 'Color', [0 0.5 0.8], 'DisplayName', 'Efficient'); % Standard in blue

% Plot Max Sharpe Points B and D
plot(VB, RB, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio B');
plot(VD, RD, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio D');
plot(VA(1), RA(1), 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio A');
plot(VC, RC, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio C');

% Add labels, legend, and title
xlabel('Volatility (\sigma)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Expected Return (\mu)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
title('Efficient vs Standard', 'FontSize', 14, 'FontWeight', 'bold');

%% Point 3
% Standard Constraints
p = Portfolio("NumAssets",NumAssets,'Name','MeanVariance');
p = setDefaultConstraints(p);

% Number of simulations
N = 500;

% Resampled Frontiers
RiskPtfSim = zeros(100,N);
RetPtfSim = zeros(100,N);

rng(10); % Fix the seed if you want to compare the results with the report

% Max Sharpe Ratio Portfolios
w_sharp_sim = zeros(NumAssets,N);
w_H_sharpe_sim = zeros(NumAssets,N);
w_min_sim = zeros(NumAssets,N);
w_min_sim_rob = zeros(NumAssets,N);

FrontierVolaR_sim = zeros(100,N);
FrontierRetR_sim = zeros(100,N);

tic
for n = 1:N
    % Gaussian parameters computation
    R = mvnrnd(ExpRet, V, length(LogRet));
    NewExpRet = mean(R);
    NewCov = cov(R);

    % Standard Frontier computation
    Psim = Portfolio("NumAssets",NumAssets);
    Psim = setDefaultConstraints(Psim);
    Psim = setAssetMoments(Psim, NewExpRet, NewCov);
    w_sim = estimateFrontier(Psim, 100);
    [RiskPtfSim(:,n), RetPtfSim(:,n)] = estimatePortMoments(Psim, w_sim);
    w_sharp_sim(:,n) = estimateMaxSharpeRatio(Psim);
    aux = Psim.estimateFrontierLimits();
    w_min_sim(:,n) = aux(:,1);

    % Efficient Frontier computation
    p = Portfolio("NumAssets",NumAssets);
    p = setDefaultConstraints(p);
    p = setAssetMoments(p,NewExpRet,NewCov);
    p = p.setInequality(A, b);
    pwgt = p.estimateFrontier(100);
    [FrontierVolaR_sim(:,n), FrontierRetR_sim(:,n)] = estimatePortMoments(p,pwgt);
    w_H_sharpe_sim(:,n) = estimateMaxSharpeRatio(p);
    aux = p.estimateFrontierLimits();
    w_min_sim_rob(:,n) = aux(:,1);
end
disp("Resampling method Computational time")
toc

% Frontiers
FrontierVola_Res = mean(RiskPtfSim,2);
FrontierRet_Res = mean(RetPtfSim,2);

% Efficient Frontiers
FrontierVolaR_Res = mean(FrontierVolaR_sim,2);
FrontierRetR_Res = mean(FrontierRetR_sim,2);

% MVP ptfs
w_E_MVP = mean(w_min_sim,2);
w_F_MVP = mean(w_min_sim_rob,2);

% Max Sharpe Ratio ptfs
w_G_sharp = mean(w_sharp_sim,2);
w_H_sharp = mean(w_H_sharpe_sim,2);

% %% Check
% sum(w_E_MVP)
% sum(w_F_MVP)
% sum(w_G_sharp)
% sum(w_H_sharp)

%% Plot
figure;
plot(FrontierVola_Res, FrontierRet_Res, 'LineWidth', 4);
hold on;
plot(FrontierVola, FrontierRet, 'LineWidth', 4);

xlabel('Risk (Volatility)');
ylabel('Return');
title('Portfolio Efficient Frontier');
legend('Resampled Frontier','Standard Frontier');

figure()
% Plot the efficient frontier
plot(FrontierVolaR_Res, FrontierRetR_Res, 'LineWidth', 4);
hold on
plot(FrontierVolaR, FrontierRetR, 'LineWidth', 4);

% Add labels and legend
xlabel('Risk (Volatility)');
ylabel('Return');
title('Portfolio Efficient Frontier with Constraints');
legend('Resampled Frontier','Standard Frontier');

%% Point 4
% Load the capitalizations
table_caps = readtable(fullfile(path_map,'capitalizations.xlsx'));

% Load the header and the data
assetNames = table_caps.Properties.VariableNames(2:end);
marketCaps = table_caps{1,2:end};

% Convert the row of data into a timetable
CapsTable = array2table(values, 'VariableNames', assetNames);

% Set the number of views 
num_views = 2;

% Compute the degree of uncertainty about the prior
tau = 1/length(ret);

% Matrix of views 
P = zeros(num_views, NumAssets);

% Expected return on each view
view_returns = zeros(num_views, 1);

% View on technology vs. financials: set that technologysector will
% outperform the fiancial sector by 2%
P(1, strcmp(assetNames, 'InformationTechnology')) = 1;
P(1, strcmp(assetNames, 'Financials')) = -1;
view_returns(1) = 0.02;

% View on the Momentum vs. Low volatilty: set that Momenum will outperform
% low volatility by 1%
P(2, strcmp(assetNames, 'Momentum')) = 1;
P(2, strcmp(assetNames, 'LowVolatility')) = -1;
view_returns(2) = 0.01;

view_returns = view_returns/252; % Daily returns

% Variance - covariance matrix of the views (assuming the vews are
% indpenedent)
Omega = zeros(num_views);
Cov_ret_matrix = cov(ret);
Omega(1,1) = tau.*P(1,:)*Cov_ret_matrix*P(1,:)';
Omega(2,2) = tau.*P(2,:)*Cov_ret_matrix*P(2,:)';
Omega = Omega/252;
% Compute each asset's capitalization in the overall market
w_caps = marketCaps(1:16)/sum(marketCaps(1:16));

% Set risk aversion coefficient
lambda = 1.2;

% Expected return at the equilibrium
% ATTENTION: w_caps is a row vector --> need to transpose it
mu_mkt = lambda.*Cov_ret_matrix*w_caps';

% Rescale by the confidence in the market data.
C = tau.*Cov_ret_matrix;

% Compute the posterior expected returns
muBL = inv(inv(C)+P'*inv(Omega)*P)*(P'*inv(Omega)*view_returns + inv(C)*mu_mkt);

% Compute the posterior covariance returns
covBL = inv(P'*inv(Omega)*P + inv(C));

% Compute the Black-Litterman portfolio
portBL = Portfolio('NumAssets', NumAssets, 'Name', 'Black-Litterman');
portBL = setDefaultConstraints(portBL);

% Add the original covariance of asset returns and the uncertainty due to
% investor views
portBL = setAssetMoments(portBL, muBL, Cov_ret_matrix+covBL);
wtsBL = estimateMaxSharpeRatio(portBL);
[risk, ~] = estimatePortMoments(portBL, wtsBL);

% Mximum sharp ratio portfolio
w_L_maxsharp = estimateMaxSharpeRatio(portBL);

% Minimum variance portfolio
[~, w_I_MVP] = portBL.estimateFrontierLimits();
w_I_MVP = w_I_MVP(:,1);

% Plot
ax1 = subplot(1,2,1);
idx = w_L_maxsharp >0.001;
pie(ax1, w_L_maxsharp(idx), assetNames(idx));
title(ax1, portBL.Name, 'Position', [-0.05, 1.6, 0]);

ax2 = subplot(1,2,2);
idx_BL = w_I_MVP > 0.001;
pie(ax2, w_I_MVP(idx_BL), assetNames(idx_BL));
title(ax2, portBL.Name, 'Position', [-0.05, 1.6, 0]);

%Plot the frontier

pwgt = portBL.estimateFrontier(100);
[FrontierVolaBL, FrontierRetBL] = estimatePortMoments(portBL,pwgt);

[VL, RL] = estimatePortMoments(portBL, w_L_maxsharp);
[VI, RI] = estimatePortMoments(portBL, w_I_MVP);

figure()
plot(FrontierVolaBL, FrontierRetBL, 'LineWidth', 2, 'Color', [0.2 0.6 0.3], 'DisplayName', 'BL frontier'); % Efficient in green
hold on

% Plot Max Sharpe Points B and D
plot(VI, RI, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio I');
plot(VL, RL, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Portfolio L');

% Add labels, legend, and title
xlabel('Volatility (\sigma)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Expected Return (\mu)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
title('Black-Litterman ', 'FontSize', 14, 'FontWeight', 'bold');


%% Point 5
% Portfolio M: Maximum Diversified Portfolio with Constraints
Aeq = ones(1, NumAssets);  % Sum of weights = 1
beq = 1;
lb = zeros(1, NumAssets);  % Lower bounds for weights
ub = ones(1, NumAssets);   % Upper bounds for weights
x0 = ones(NumAssets, 1) / NumAssets;  % Initialization: equally distributed weights

% Additional constraints
% 1. Total exposure to cyclical sectors > 20%
CyclicalIdx = [2, 4, 6, 10, 11];   % Indexes of cyclical sectors
I = eye(NumAssets);
A_cyclicals = -sum(I(CyclicalIdx, :), 1);
b_cyclicals = -0.20;

% 2. Sum of absolute differences between optimal weights and benchmark > 20%
Benchmark = w_caps;  % Benchmark weights

% Function for the sum of absolute differences
total_difference = @(benchmark, w) sum(abs(benchmark - w));

% Nonlinear constraint function
nonlincon = @(w) deal(0.20 - total_difference(Benchmark', w), []);  % Inequality constraints

% Combining constraints
A = A_cyclicals;
b = b_cyclicals;

options = optimoptions('fmincon','MaxFunctionEvaluations',1e5);

% Optimization of the Diversification Ratio
[w_M_MDR, fval] = fmincon(@(x) -getDiversificationRatio(x, LogRet), x0, A, b, Aeq, beq, lb, ub, nonlincon, options);

MaxDR = -fval;  % Maximum Diversification Ratio
[relRC_MDR, RC_MDR, mVol_MDR] = getRiskContributions(w_M_MDR, LogRet);


% Relative Risk Contributions
figure()
threshold = 0.01; % Soglia per includere le categorie
idx = relRC_MDR > threshold;
pie(relRC_MDR(idx), assetNames(idx));
title('Relative Risk Contributions for Maximum Diversified Portfolio (MDR)');

% Portfolio Weights
figure()
idx = w_M_MDR > threshold;
pie(w_M_MDR(idx), assetNames(idx));
title('Weights for Maximum Diversified Portfolio (MDR)');


%% Portfolio N: Maximum Entropy Portfolio (Volatility Contributions) with Constraints

% Optimization of entropy in volatility contributions
w_N_MaxEntropyVol = fmincon(@(x) -getEntropy(getVolContributions(x, LogRet)), x0, A, b, Aeq, beq, lb, ub, nonlincon, options);
MaxEntropy_Vol = getEntropy(w_N_MaxEntropyVol);


% Portfolio Weights
figure();
threshold = 0.01; 
idx = w_N_MaxEntropyVol > threshold;
pie(w_N_MaxEntropyVol(idx), assetNames(idx));
title('Weights for Maximum Entropy Portfolio');

% Volatility Contributions
figure();
volContr = getVolContributions(w_N_MaxEntropyVol, LogRet);
idx = volContr > threshold;
pie(volContr(idx), assetNames(idx));
title('Volatility Contributions for Maximum Entropy Portfolio');


%% Point 6
RetStd = (LogRet-ExpRet)./ std(LogRet);
[~, ~, latent, ~, explained, ~] = pca(RetStd);

TotVar = sum(latent);
ExplainedVar = latent/TotVar;
n_list = 1:16;
CumExplainedVar = zeros(1, size(n_list,2));
for i = 1:size(n_list,2)
    n = n_list(i);
    CumExplainedVar(1,i) = getCumulativeExplainedVar(latent, n);
end

% Plot
% Plot 1: Percentage of Explained Variances for each Principal Component
h = figure();
bar(n_list, ExplainedVar, 'FaceColor', [0 0.5 0.8]); % Light blue bars
grid on;
title('Percentage of Explained Variance for Each Principal Component', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Principal Components', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Percentage of Explained Variance (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10, 'LineWidth', 1); % Adjust axis properties
set(gcf, 'Position', [100, 100, 800, 600]); % Adjust figure size

% Plot 2: Total Percentage of Explained Variances for the first n-components
f = figure();
plot(n_list, CumExplainedVar, '-m', 'LineWidth', 2); % Magenta line with width
hold on;
scatter(n_list, CumExplainedVar, 50, 'm', 'filled'); % Magenta filled circles
grid on;
title('Total Percentage of Explained Variance for the First n-Components', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Number of Principal Components', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cumulative Explained Variance (%)', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 10, 'LineWidth', 1); % Adjust axis properties
set(gcf, 'Position', [120, 120, 800, 600]); % Adjust figure size
legend({'Cumulative Variance', 'Data Points'}, 'Location', 'southeast', 'FontSize', 10);
hold off;

%%
% LogReturn standardization
std_Ret = std(LogRet);
RetStd = (LogRet-ExpRet)./ std_Ret ;

indices = find(cumsum(explained) >= 90);
k = indices(1);

% Compute PCA using num_factors principal components
[factorLoading, factorRetn, latent, r, explained, mu] = pca(RetStd,'NumComponents', k);

targetRisk = 0.7;  % Standard deviation of portfolio return
tRisk = targetRisk*targetRisk;  % Variance of portfolio return
meanStockRetn = ExpRet;
p = length(nm);

covarFactor=cov(factorRetn);
reconReturn=std_Ret.*(factorRetn*factorLoading')+ExpRet;
unexplainedRetn=LogRet-reconReturn;
unexplainedCovar=diag(cov(unexplainedRetn));
D=diag(unexplainedCovar);
covarAsset=factorLoading*covarFactor*factorLoading'+D;

% Expected return maximization:
%fun = @(x) -(ExpRet*x-((factorLoading'*x)' * covarFactor * (factorLoading'*x) + x'*D*x));
fun = @(x) -(ExpRet*x);

% Optimization definition
optimProb = optimproblem('Description','Portfolio with factor covariance matrix','ObjectiveSense','max');
wgtAsset = optimvar('asset_weight', p, 1, 'Type', 'continuous', 'LowerBound', 0, 'UpperBound', 1);
wgtFactor = optimvar('factor_weight', k, 1, 'Type', 'continuous');

optimProb.Objective = sum(meanStockRetn'.*wgtAsset);

% Constraints
optimProb.Constraints.asset_factor_weight = factorLoading'*wgtAsset - wgtFactor == 0;
optimProb.Constraints.risk = wgtFactor'*covarFactor*wgtFactor + wgtAsset'*D*wgtAsset <= tRisk;
optimProb.Constraints.budget = sum(wgtAsset) == 1;

x0 = struct('asset_weight', ones(p, 1)/p, 'factor_weight', zeros(k, 1));

% Optimization
opt = optimoptions("fmincon", "Algorithm","sqp", "Display", "off",'ConstraintTolerance', 1.0e-8, 'OptimalityTolerance', 1.0e-8, 'StepTolerance', 1.0e-8);
x = solve(optimProb,x0, "Options",opt);
w_P_PCA = x.asset_weight;


%% Point 7
% Var-modified Portfolio optimization
weights_EW = ones(size(prices_val, 2), 1).*1/size(prices_val,2);
pRet = weights_EW'*LogRet';
ConfLevel = [0.95, 0.99];

% Var Computation
VaR_95 = quantile(pRet, 1-ConfLevel(1,1));
VaR_99 = quantile(pRet, 1-ConfLevel(1,2));

% Var-Modified Sharpe Ratio (Var instead of Volatility as risk factor)
z_alpha = icdf('normal',0.95,0,1);
objective = @(w) -((ExpRet*w) / (ExpRet*w+z_alpha*sqrt(w'*Cov_ret_matrix*w)));

% Constraints
Aeq = ones(1, NumAssets);
beq = 1;
lb = zeros(NumAssets, 1);
ub = ones(NumAssets, 1);
x0 = rand(1,NumAssets)';
x0 = x0/sum(x0);

% Portfolio optimization
w_Q_opt = fmincon(objective, x0, [], [], Aeq, beq, lb, ub);

[VQ, RQ] = estimatePortMoments(Port, w_Q_opt);
figure();

% Plot the Efficient Frontier
plot(FrontierVola, FrontierRet, 'LineWidth', 2, 'Color', [0 0.5 0.8], 'DisplayName', 'Standard'); % Blue-green line

hold on;

% Plot Maximum Sharpe Ratio Portfolios B and Q
plot(VB, RB, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k', 'DisplayName', 'MaxSharpeB'); % Blue filled marker
plot(VQ, RQ, 'o', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'MaxSharpeQ'); % Red filled marker

% Add labels and legend
xlabel('Volatility (\sigma)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Expected Return (\mu)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);

%% RESULTS DISPLAY
SectorNames = {'Information', 'Financials', 'HealthCare', 'Consumer', 'Communic', 'Industrials', 'ConsumerS', 'Energy', 'Utilities', 'RealEstate', 'Materials', 'Momentum', 'Value', 'Growth','Quality','LowVolatility'};
% Combine weights for all portfolios
WeightsMatrix = [w_A_MVP(:,1), w_B_sharp(:,1), w_C_MVP(:,1), w_D_sharp(:,1), ...
                 w_E_MVP(:,1), w_F_MVP(:,1), w_G_sharp(:,1), w_H_sharp(:,1), ...
                 w_I_MVP(:,1), w_L_maxsharp(:,1), w_M_MDR(:,1), w_N_MaxEntropyVol(:,1), ...
                 w_P_PCA(:,1), w_Q_opt(:,1)];

% Create an array of portfolio names
PortfolioNames = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'Q'};
WeightsMatrix(WeightsMatrix < 0.0001) = 0;
% Transpose the matrix and create a table
TransposedTable = array2table(WeightsMatrix, 'RowNames', SectorNames, 'VariableNames', PortfolioNames);

% Display the transposed table
disp(TransposedTable);

%% Point 8
% Return definition
ret = prices_val(2:end,:)./prices_val(1:end-1,:);

% Metrics Computation
[annRet_A, annVol_A, Sharpe_A, MaxDD_A, Calmar_A] = metrics(ret,w_A_MVP(:,1));
[annRet_B, annVol_B, Sharpe_B, MaxDD_B, Calmar_B] = metrics(ret,w_B_sharp(:,1));
[annRet_C, annVol_C, Sharpe_C, MaxDD_C, Calmar_C] = metrics(ret,w_C_MVP(:,1));
[annRet_D, annVol_D, Sharpe_D, MaxDD_D, Calmar_D] = metrics(ret,w_D_sharp(:,1));
[annRet_E, annVol_E, Sharpe_E, MaxDD_E, Calmar_E] = metrics(ret,w_E_MVP(:,1));
[annRet_F, annVol_F, Sharpe_F, MaxDD_F, Calmar_F] = metrics(ret,w_F_MVP(:,1));
[annRet_G, annVol_G, Sharpe_G, MaxDD_G, Calmar_G] = metrics(ret,w_G_sharp(:,1));
[annRet_H, annVol_H, Sharpe_H, MaxDD_H, Calmar_H] = metrics(ret,w_H_sharp(:,1));
[annRet_I, annVol_I, Sharpe_I, MaxDD_I, Calmar_I] = metrics(ret,w_I_MVP(:,1));
[annRet_L, annVol_L, Sharpe_L, MaxDD_L, Calmar_L] = metrics(ret,w_L_maxsharp(:,1));
[annRet_M, annVol_M, Sharpe_M, MaxDD_M, Calmar_M] = metrics(ret,w_M_MDR(:,1));
[annRet_N, annVol_N, Sharpe_N, MaxDD_N, Calmar_N] = metrics(ret,w_N_MaxEntropyVol(:,1));
[annRet_P, annVol_P, Sharpe_P, MaxDD_P, Calmar_P] = metrics(ret,w_P_PCA(:,1));
[annRet_Q, annVol_Q, Sharpe_Q, MaxDD_Q, Calmar_Q] = metrics(ret,w_Q_opt(:,1));


PortfolioNames = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'Q'}';
AnnualReturns = [annRet_A, annRet_B, annRet_C, annRet_D, annRet_E, annRet_F, annRet_G, ...
                 annRet_H, annRet_I, annRet_L, annRet_M, annRet_N, annRet_P, annRet_Q]';
AnnualVolatility = [annVol_A, annVol_B, annVol_C, annVol_D, annVol_E, annVol_F, annVol_G, ...
                    annVol_H, annVol_I, annVol_L, annVol_M, annVol_N, annVol_P, annVol_Q]';
SharpeRatios = [Sharpe_A, Sharpe_B, Sharpe_C, Sharpe_D, Sharpe_E, Sharpe_F, Sharpe_G, ...
                Sharpe_H, Sharpe_I, Sharpe_L, Sharpe_M, Sharpe_N, Sharpe_P, Sharpe_Q]';
MaxDrawdowns = [MaxDD_A, MaxDD_B, MaxDD_C, MaxDD_D, MaxDD_E, MaxDD_F, MaxDD_G, ...
                MaxDD_H, MaxDD_I, MaxDD_L, MaxDD_M, MaxDD_N, MaxDD_P, MaxDD_Q]';
CalmarRatios = [Calmar_A, Calmar_B, Calmar_C, Calmar_D, Calmar_E, Calmar_F, Calmar_G, ...
                Calmar_H, Calmar_I, Calmar_L, Calmar_M, Calmar_N, Calmar_P, Calmar_Q]';

% Combine into a table
PerformanceTable = table(PortfolioNames, AnnualReturns, AnnualVolatility, SharpeRatios, MaxDrawdowns, CalmarRatios);

% Display the table
disp(PerformanceTable);

%plot
w_ptf = [w_A_MVP(:,1) w_B_sharp w_C_MVP(:,1) w_D_sharp w_E_MVP w_F_MVP w_G_sharp w_H_sharp w_I_MVP w_L_maxsharp w_M_MDR w_N_MaxEntropyVol w_P_PCA w_Q_opt weights_EW]
total_plot(ret, w_ptf,dates_)

%% Part B
% Selection of a subset of Dates
start_dt_B = datetime('01/01/2024', 'InputFormat', 'dd/MM/yyyy'); 
end_dt_B = datetime('25/10/2024', 'InputFormat', 'dd/MM/yyyy');

rng_dates_B = timerange(start_dt_B, end_dt_B, 'closed');
subsample_B = myPrice_dt(rng_dates_B,:);

prices_val_B = subsample_B.Variables;
dates_B = subsample_B.Time;

%% New Metrics
ret_B = prices_val_B(2:end,:)./prices_val_B(1:end-1,:);
[annRet_EW_B, annVol_EW_B, Sharpe_EW_B, MaxDD_EW_B, Calmar_EW_B] = metrics(ret_B,weights_EW(:,1));
[annRet_A_B, annVol_A_B, Sharpe_A_B, MaxDD_A_B, Calmar_A_B] = metrics(ret_B,w_A_MVP(:,1));
[annRet_B_B, annVol_B_B, Sharpe_B_B, MaxDD_B_B, Calmar_B_B] = metrics(ret_B,w_B_sharp(:,1));
[annRet_C_B, annVol_C_B, Sharpe_C_B, MaxDD_C_B, Calmar_C_B] = metrics(ret_B,w_C_MVP(:,1));
[annRet_D_B, annVol_D_B, Sharpe_D_B, MaxDD_D_B, Calmar_D_B] = metrics(ret_B,w_D_sharp(:,1));
[annRet_E_B, annVol_E_B, Sharpe_E_B, MaxDD_E_B, Calmar_E_B] = metrics(ret_B,w_E_MVP(:,1));
[annRet_F_B, annVol_F_B, Sharpe_F_B, MaxDD_F_B, Calmar_F_B] = metrics(ret_B,w_F_MVP(:,1));
[annRet_G_B, annVol_G_B, Sharpe_G_B, MaxDD_G_B, Calmar_G_B] = metrics(ret_B,w_G_sharp(:,1));
[annRet_H_B, annVol_H_B, Sharpe_H_B, MaxDD_H_B, Calmar_H_B] = metrics(ret_B,w_H_sharp(:,1));
[annRet_I_B, annVol_I_B, Sharpe_I_B, MaxDD_I_B, Calmar_I_B] = metrics(ret_B,w_I_MVP(:,1));
[annRet_L_B, annVol_L_B, Sharpe_L_B, MaxDD_L_B, Calmar_L_B] = metrics(ret_B,w_L_maxsharp(:,1));
[annRet_M_B, annVol_M_B, Sharpe_M_B, MaxDD_M_B, Calmar_M_B] = metrics(ret_B,w_M_MDR(:,1));
[annRet_N_B, annVol_N_B, Sharpe_N_B, MaxDD_N_B, Calmar_N_B] = metrics(ret_B,w_N_MaxEntropyVol(:,1));
[annret_P_B, annVol_P_B, Sharpe_P_B, MaxDD_P_B, Calmar_P_B] = metrics(ret_B,w_P_PCA(:,1));
[annRet_Q_B, annVol_Q_B, Sharpe_Q_B, MaxDD_Q_B, Calmar_Q_B] = metrics(ret_B,w_Q_opt(:,1));

%plot
w_ptf = [w_A_MVP(:,1) w_B_sharp w_C_MVP(:,1) w_D_sharp w_E_MVP w_F_MVP w_G_sharp w_H_sharp w_I_MVP w_L_maxsharp w_M_MDR w_N_MaxEntropyVol w_P_PCA w_Q_opt weights_EW]
total_plot(ret_B, w_ptf,dates_B)

%%
AnnualReturns_B = [annRet_EW_B;annRet_A_B; annRet_B_B; annRet_C_B; annRet_D_B; annRet_E_B; ...
                   annRet_F_B; annRet_G_B; annRet_H_B; annRet_I_B; annRet_L_B; ...
                   annRet_M_B; annRet_N_B; annret_P_B; annRet_Q_B];

AnnualVolatility_B = [annVol_EW_B;annVol_A_B; annVol_B_B; annVol_C_B; annVol_D_B; annVol_E_B; ...
                      annVol_F_B; annVol_G_B; annVol_H_B; annVol_I_B; annVol_L_B; ...
                      annVol_M_B; annVol_N_B; annVol_P_B; annVol_Q_B];

SharpeRatios_B = [Sharpe_EW_B;Sharpe_A_B; Sharpe_B_B; Sharpe_C_B; Sharpe_D_B; Sharpe_E_B; ...
                  Sharpe_F_B; Sharpe_G_B; Sharpe_H_B; Sharpe_I_B; Sharpe_L_B; ...
                  Sharpe_M_B; Sharpe_N_B; Sharpe_P_B; Sharpe_Q_B];

MaxDrawdowns_B = [MaxDD_EW_B;MaxDD_A_B; MaxDD_B_B; MaxDD_C_B; MaxDD_D_B; MaxDD_E_B; ...
                  MaxDD_F_B; MaxDD_G_B; MaxDD_H_B; MaxDD_I_B; MaxDD_L_B; ...
                  MaxDD_M_B; MaxDD_N_B; MaxDD_P_B; MaxDD_Q_B];

CalmarRatios_B = [Calmar_EW_B;Calmar_A_B; Calmar_B_B; Calmar_C_B; Calmar_D_B; Calmar_E_B; ...
                  Calmar_F_B; Calmar_G_B; Calmar_H_B; Calmar_I_B; Calmar_L_B; ...
                  Calmar_M_B; Calmar_N_B; Calmar_P_B; Calmar_Q_B];

% Create an array of portfolio names
PortfolioNames_B = {'EW','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'Q'}';

% Combine all metrics into a table
MetricsTable_B = table(PortfolioNames_B, AnnualReturns_B, AnnualVolatility_B, ...
                       SharpeRatios_B, MaxDrawdowns_B, CalmarRatios_B, ...
                       'VariableNames', {'Portfolio', 'AnnualReturn', 'AnnualVolatility', ...
                                         'SharpeRatio', 'MaxDrawdown', 'CalmarRatio'});

% Display the table
disp(MetricsTable_B);

%% Portfolios 
figure()
plot(dates_B, prices_val_B*w_A_MVP(:,1))
hold on
plot(dates_B, prices_val_B*w_B_sharp)
plot(dates_B, prices_val_B*w_C_MVP(:,1))
plot(dates_B, prices_val_B*w_D_sharp)
plot(dates_B, prices_val_B*w_E_MVP)
plot(dates_B, prices_val_B*w_F_MVP)
plot(dates_B, prices_val_B*w_G_sharp)
plot(dates_B, prices_val_B*w_H_sharp)
plot(dates_B, prices_val_B*w_I_MVP)
plot(dates_B, prices_val_B*w_L_maxsharp)
plot(dates_B, prices_val_B*w_M_MDR)
plot(dates_B, prices_val_B*w_N_MaxEntropyVol)
plot(dates_B, prices_val_B*w_P_PCA)
plot(dates_B, prices_val_B*w_Q_opt)
legend('A','B','C','D','E','F','G','H','I','L','M','N','P','Q')
