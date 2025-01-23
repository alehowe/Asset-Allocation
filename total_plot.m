function total_plot(ret, w_ptf,date)

% This function plots standardized paths of a set of portfolios during the
% analyzed period. All the portfolios starts from 100

figure();

for i=1:size(w_ptf,2)
    equity=[];
    equity=cumprod(ret*w_ptf(:,i));
    equity=100.*equity/equity(1);
    plot(date(2:end), equity, 'LineWidth',2)
    hold on
end
xlabel('Date')
ylabel('Portfolio Value')
legend('A' ,'B', 'C', 'D' ,'E' ,'F' ,'G' ,'H' ,'I' ,'L' ,'M' ,'N' ,'P', 'Q','EW', 'Location','eastoutside')
ylim([60 160]);
end