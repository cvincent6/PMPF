%%Colin Vincent
%%Machine Learning Project
%%Power Market Price Forecasting
%%October 27, 2017

%% Example Code For ARIMA (Autoregressive Integrated Moving Average)

%Forecast Multiplicative ARIMA Model
%https://www.mathworks.com/help/econ/forecast-airline-passenger-counts.html

load(fullfile(matlabroot,'examples','econ','Data_Airline.mat'))
y = log(Data);
T = length(y);

Mdl = arima('Constant',0,'D',1,'Seasonality',12,...
    'MALags',1,'SMALags',12);
EstMdl = estimate(Mdl,y);

[yF,yMSE] = forecast(EstMdl,60,'Y0',y);
upper = yF + 1.96*sqrt(yMSE);
lower = yF - 1.96*sqrt(yMSE);

figure
plot(y,'Color',[.75,.75,.75])
hold on
h1 = plot(T+1:T+60,yF,'r','LineWidth',2);
h2 = plot(T+1:T+60,upper,'k--','LineWidth',1.5);
plot(T+1:T+60,lower,'k--','LineWidth',1.5)
xlim([0,T+60])
title('Forecast and 95% Forecast Interval')
legend([h1,h2],'Forecast','95% Interval','Location','NorthWest')
hold off


rng 'default';
res = infer(EstMdl,y);
Ysim = simulate(EstMdl,60,'NumPaths',500,'Y0',y,'E0',res);

yBar = mean(Ysim,2);
simU = prctile(Ysim,97.5,2);
simL = prctile(Ysim,2.5,2);

figure
h1 = plot(yF,'Color',[.85,.85,.85],'LineWidth',5);
hold on
h2 = plot(yBar,'k--','LineWidth',1.5);
xlim([0,60])
plot([upper,lower],'Color',[.85,.85,.85],'LineWidth',5)
plot([simU,simL],'k--','LineWidth',1.5)
title('Comparison of MMSE and Monte Carlo Forecasts')
legend([h1,h2],'MMSE','Monte Carlo','Location','NorthWest')
hold off
