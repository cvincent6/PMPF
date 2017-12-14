%   Drew HasBrouck
%   EECE 5644 Final Project
%   Power Market Price Forecasting
%   Main Module

clear;
clc;

%   Read in historical electricity price data
trainData = csvread('NYC-dataset_TRAINING_(2014).csv');
%trainData = csvread('NYC-dataset 3 years.csv');
s = size(trainData);
numPoints = s(1,1);
for i=1:numPoints
    trainData(i,1) = i;
end

%   Statistically determine price spikes
spikeAvgPeriod = 30*24;
priceArray = zeros(3);
numSpikes = 0;
numPossibleSpikes = numPoints-spikeAvgPeriod;

priceArray(1,1) = trainData(1,1);
priceArray(1,2) = trainData(1,2);
mu = 60;
sigma = 60;
j = 2;
for i=2:spikeAvgPeriod
    priceArray(j,1) = trainData(i,1);
    priceArray(j,2) = trainData(i,2);
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

% plot(trainData(:,1),trainData(:,2),'color','black');
% hold on;
% scatter(spikePriceArray(:,1),spikePriceArray(:,2),10,'red');
% hold on;
% plot(priceArray(:,1),priceArray(:,4),'color','blue');
% hold on;
% plot(priceArray(:,1),priceArray(:,5),'color','green');
% xlabel('Hour');
% ylabel('Price ($/MWh)');
% title('Hourly Power Prices w/ Spikes Identified');

%   Wavelet Transform
for i=1:s(1,1)
    priceArrayWav(i,1) = 1;
    sig = priceArray(:,2);
    [C,L] = wavedec(sig,3,'db5');
    cd1 = detcoef(C,L,1);
    ca3 = detcoef(C,L,3);
    %priceArrayWav(i,2:end) = dbwavf('db5');
    g = 1;
   
end

%   ARIMA For All WT Subseries

%   Classification
%       Develop Candidates for Classifiers
n = 200;
d = 4;
%matrix x of nx(d+1) dimension containing candidates [use +1 for class]

%       RVM
%       DT
%       PNN
%implement PNN from paper here

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
    
%       Classifier Vote

%   Normal Price Time Series Construction

%   Price Spike Series Construction


