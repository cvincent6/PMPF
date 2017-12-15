%   Drew HasBrouck
%   EECE 5644 Final Project
%   Power Market Price Forecasting
%   Main Module

clear;
clc;

%hour for prediction
for h = 8761:17520

%   Read in historical electricity price data
%trainData = csvread('NYC-dataset_TRAINING_(2014).csv');
trainData = csvread('NYC-dataset 3 years.csv');
s = size(trainData);
numPoints = s(1,1);
for i=1:numPoints
    trainData(i,1) = i;
end

%   Statistically determine price spikes
spikeAvgPeriod = 30*24;
priceArray = zeros(6);
numSpikes = 0;
numPossibleSpikes = numPoints;

priceArray(1,1) = trainData(1,1);
priceArray(1,2) = trainData(1,2);
priceArray(1,6) = trainData(1,3);
mu = 60;
sigma = 60;
j = 2;
for i=2:spikeAvgPeriod
    priceArray(j,1) = trainData(i,1);
    priceArray(j,2) = trainData(i,2);
    priceArray(j,6) = trainData(i,3);
    mu = mean(trainData(1:(i-1),2));
    sigma = sqrt(var(trainData(1:(i-1),2)));
%     if mu>75
%         mu = 75;
%     end
%     if sigma>40
%         sigma = 40;
%     end
    priceArray(j,4) = mu;
    priceArray(j,5) = sigma;
    thresh = mu+3*sigma;
    if trainData(i,2) > thresh
        priceArray(j,3) = 1;
        numSpikes = numSpikes+1;
    else
        priceArray(j,3) = 0;
    end
    j = j+1;
end
for i=(spikeAvgPeriod+1):numPoints
    priceArray(j,1) = trainData(i,1);
    priceArray(j,2) = trainData(i,2);
    priceArray(j,6) = trainData(i,3);
    mu = mean(trainData((i-spikeAvgPeriod):(i-1),2));
    sigma = sqrt(var(trainData((i-spikeAvgPeriod):(i-1),2)));
%      if mu>75
%         mu = 75;
%     end
%     if sigma>40
%         sigma = 40;
%     end
    priceArray(j,4) = mu;
    priceArray(j,5) = sigma;
    thresh = mu+3*sigma;
    if trainData(i,2) > thresh
        priceArray(j,3) = 1;
        numSpikes = numSpikes+1;
    else
        priceArray(j,3) = 0;
    end
    j = j+1;
end

s = size(priceArray);
j = 1;
k = 1;
for i=1:s(1,1)
    if priceArray(i,3) == 0
        normalPriceArray(j,:) = priceArray(i,:);
        j = j+1;
    else
        spikePriceArray(k,:) = priceArray(i,:);
        k = k+1;
    end
end

meanN = mean(normalPriceArray(:,2));
sigmaN = sqrt(var(normalPriceArray(:,2)));
meanS = mean(spikePriceArray(:,2));
sigmaS = sqrt(var(spikePriceArray(:,2)));

%Plot statistical spike analysis
% plot(trainData(:,1),trainData(:,2),'color','black');
% hold on;
% scatter(spikePriceArray(:,1),spikePriceArray(:,2),10,'red');
% hold on;
% plot(priceArray(:,1),priceArray(:,4),'color','blue');
% hold on;
% plot(priceArray(:,1),priceArray(:,5),'color','green');
% xlabel('Hour');
% ylabel('Price ($/MWh)');
% title('Hourly Power Prices w/ Spikes Identified (2014-2016)');
% legend on;
% legend('Power Price','Price Spike','30 Day Mean','30 Day Sigma');

%   Wavelet Transform
sig = priceArray(h-8760:h-1,2);
[C,L] = wavedec(sig,3,'db5');
cd1 = detcoef(C,L,1);
ca3 = detcoef(C,L,3);
j = 1;
count = 1;
for i=h-8760:h-1
    cd1e(i,1) = cd1(j,1);
    if count == 2
        count = 0;
        j = j+1;
    end
    count = count+1;
end
j = 1;
count = 1;
for i=h-8760:h-1
    ca3e(i,1) = ca3(j,1);
    if count == 8
        count = 0;
        j = j+1;
    end
    count = count+1;
end

%Plot wavelet components
% plot(priceArray(:,1),sig,'color','black');
% hold on;
% plot(priceArray(:,1),ca3e,'color','green');
% hold on;
% plot(priceArray(:,1),cd1e,'color','red');

%   ARIMA For All WT Subseries
%Price
% priceA = arima('Seasonality',2,'D',5);%,'SAR',[.25,.5],'SMA',[.25,.5]);
% priceA = arima(4,1,3);
% priceA = estimate(priceA,priceArray(:,2));
% [Y,YMSE] = forecast(priceA,24);
% prediction(1:24,1) = Y(1:24,1);


%   Classification
%       Develop Candidates for Classifiers
candidates = zeros(200,5);
%Prices
candidates(:,1) = priceArray(h-200:h-1,2);
%Load
candidates(:,2) = priceArray(h-200:h-1,6);
%A3
candidates(:,3) = ca3e(h-200:h-1,1);
%D1
candidates(:,4) = cd1e(h-200:h-1,1);
%Spike 0 or 1
candidates(:,5) = priceArray(h-200:h-1,3);

n = 200;
d = 4;

%       RVM
%       DT
%       PNN
prediction(h,1) = trainData(h,2);
prediction(h,2) = trainData(h,3);
prediction(h,3) = ca3e(h-24,1);
prediction(h,4) = cd1e(h-24,1);
p = 1;
sigmaPNN = 10;
%form spike candidates
predictionNorm = prediction;
predictionNorm(:,1) = predictionNorm(:,1)/50;
predictionNorm(:,2) = predictionNorm(:,2)/6000;
candidates(:,1) = candidates(:,1)/50;
candidates(:,2) = candidates(:,2)/6000;
count = 1;
spikeCan = [];
for i=1:200
    if candidates(i,5) == 1
        spikeCan(count,:) = candidates(i,:);
        count = count+1;
    end
end
s = size(spikeCan);
m = s(1,1);
sum1 = 0;
if m>0
    for i=1:m
        adder = -(predictionNorm(h,1:4)-spikeCan(i,1:4))*transpose((predictionNorm(h,1:4)-spikeCan(i,1:4)));
        sum1 = sum1 + exp(-(predictionNorm(h,1:4)-spikeCan(i,1:4))*transpose((predictionNorm(h,1:4)-spikeCan(i,1:4)))/(2*sigmaPNN^2));
    end
end
if m>0
    probSpike = 1/((2*pi)^(p/2)*sigmaPNN^p)*1/m*sum1;
else
    probSpike = 0;
end

%form normal candidates
count = 1;
for i=1:200
    if candidates(i,5) == 0
        normCan(count,:) = candidates(i,:);
        count = count+1;
    end
end
s = size(normCan);
m = s(1,1);
sum1 = 0;
for i=1:m
    sum1 = sum1 + exp(-(predictionNorm(h,1:4)-normCan(i,1:4))*transpose((predictionNorm(h,1:4)-normCan(i,1:4)))/(2*sigmaPNN^2));
end
probNorm = 1/((2*pi)^(p/2)*sigmaPNN^p)*1/m*sum1;

%class decision
if probSpike >= probNorm
    prediction(h,5) = 1;
else
    prediction(h,5) = 0;
end
prediction(h,6) = priceArray(h,3);
prediction(h,7) = probSpike;
prediction(h,8) = probNorm;
end

%Plot spike prediction
plot(trainData(8761:9000,1),trainData(8761:9000,2),'color','black');
hold on;
scatter(trainData(8761:9000,1),prediction(8761:9000,6).*prediction(8761:9000,1),10,'red');
hold on;
scatter(trainData(8761:9000,1),prediction(8761:9000,5).*prediction(8761:9000,1),7,'blue');
xlabel('Hour');
ylabel('Price ($/MWh)');
title('Hourly Power Prices w/ Spikes Classified (2014)');
legend on;
legend('Power Price','Price Spike (Statistical)','Price Spike (Classifier)');

% %Train
% norm = 0;
% w = zeros(n,d);
% a = zeros(n,2);
% for j=1:n
%     for k=1:d
%         for i=1:d
%             norm = norm + x(j,i)^2;
%         end
%         norm = sqrt(norm);
%         xn = x(j,k)/norm;
%         w(j,k) = xn;
%     end
%     if x(j,d+1) == 0
%         %Normals
%         a(j,1) = 1;
%     else
%         %Spikes
%         a(j,2) = 1;
%     end
% end
% 
% %Classify
% %set this parameter! (sigma)
% sigma = 2;
% for k=1:n
%     %X is sample to be classified
%     netk = transpose(w(k,:))*X;
%     if a(k,1) == 1
%         g(1) = g(1) + exp((netk-1)/sigma^2);
%     else
%         g(2) = g(2) + exp((netk-1)/sigma^2);
%     end
% end
% 
% if g(2)>g(1)
%     %figure out where to record this
%     spike = 1;
% end