function [fitresult, gof] = createFit(y1, x1)
%CREATEFIT(Y1,X1)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : y1
%      Y Output: x1
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( y1, x1 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, 'Normalize', 'on' );

% Create a figure for the plots.
figure( 'Name', 'untitled fit 1' );

% Plot fit with data.
% subplot( 2, 1, 1 );
h = plot( fitresult, xData, yData );
legend( h, 'x1 vs. y1', 'untitled fit 1', 'Location', 'NorthEast' );
% Label axes
xlabel y1
ylabel x1
set(gca,'Ydir','reverse')
grid on

% Plot residuals.
% subplot( 2, 1, 2 );
% h = plot( fitresult, xData, yData, 'residuals' );
% legend( h, 'untitled fit 1 - residuals', 'Zero Line', 'Location', 'NorthEast' );
% % Label axes
% xlabel y1
% ylabel x1
% grid on


