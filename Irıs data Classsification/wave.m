clc;
clear;
close all;
%%
load fisheriris.mat 

classnames = unique(species);
for i=1:3
    class(strcmp(species, classnames{i})) = i;
end
class = class';
tbl = [meas class];

[m, n] = size(tbl);
[~, idx] = sort(rand(m,1));

for c = 1:m
    tbl(c,1:5) = tbl(idx(c),1:5);
end

%%
numObservations = size(tbl,1);
numObservationsTrain = floor(0.70*numObservations);
numObservationsTest = numObservations - numObservationsTrain;

idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxTest = idx(numObservationsTrain+1:end);

trainn = tbl(idxTrain,1:4);

test = tbl(idxTest,1:4);
y_train = tbl(idxTrain,5);

y_test = tbl(idxTest,5);


colmn1_train=trainn(:,1);
colmn2_train=trainn(:,2);
colmn3_train=trainn(:,3);
colmn4_train=trainn(:,4);


colmn1_norm_train=colmn1_train/255;
colmn2_norm_train=colmn2_train/255;
colmn3_norm_train=colmn3_train/255;
colmn4_norm_train=colmn4_train/255;

x_train = [colmn1_norm_train,colmn2_norm_train,colmn3_norm_train,colmn4_norm_train];

colmn1_test=test(:,1);
colmn2_test=test(:,2);
colmn3_test=test(:,3);
colmn4_test=test(:,4);

colmn1_norm_test=colmn1_test/255;
colmn2_norm_test=colmn2_test/255;
colmn3_norm_test=colmn3_test/255;
colmn4_norm_test=colmn4_test/255;

x_test = [colmn1_norm_test,colmn2_norm_test,colmn3_norm_test,colmn4_norm_test];


% build model and train
rng(1)
template = templateSVM(...
    'KernelFunction','polynomial', ...
    'PolynomialOrder',3, ...
    'KernelScale','auto', ...
    'BoxConstraint',1, ...
    'Standardize',true);

model = fitcecoc(...
    x_train, ...
    y_train, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames',{'1','2','3'});

% prediction labels
predLabels = predict(model,x_test);
predLabels = str2double(predLabels);
correctPredictions=[length(predLabels)];
for i=1:length(predLabels)
    if predLabels(i)==y_test(i);
        correctPredictions(i)=1;
    else
        correctPredictions(i)=0;
    end
end

testAccuracy = sum(correctPredictions)/length(y_test)*100
[confmatCVDFT,grouporder] = confusionmat(y_test,predLabels);
figure
cm = confusionchart(y_test,predLabels);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Flower Irıs Confusion Matrix';
dftTable = helperPrecisionRecall(confmatCVDFT);
disp(dftTable)


