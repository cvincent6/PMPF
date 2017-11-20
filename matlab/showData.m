%%Colin Vincent
%%Nov 16, 2017
%%Machine Learning

%% Reading in datasets

clear all
close all
clc

% coal,fuel, oil, ...
Data = csvread('MarketData/RealDataPT.csv',1,2);
[rows,cols] = size(Data);

Data_hourly = [];

%Averaging per hour
for i = 1:rows/4
    Data_hourly(i,:) = mean(Data(i*4-3:i*4,cols));
end

figure
for k = 1:cols
    plot(Data_hourly(1:24*28,k),'-.')
    hold on
end

for x = 1:28
   plot([x*24 x*24],[0 9000],'k')
   hold on
end