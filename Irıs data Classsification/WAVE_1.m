clc;
clear;
close all;

%get data
load fisheriris.mat 
classnames = unique(species);
for i=1:3
    class(strcmp(species, classnames{i})) = i;
end
class = class';
%inputs data
data = [class meas];
% sort data
[m, n] = size(data);
[~, idx] = sort(rand(m,1));


%% cross-validation: KFold (k = 3);

%train
colmn1=data(:,1);
colmn2=data(:,2);
colmn3=data(:,3);
colmn4=data(:,4);
colmn5=data(:,5);


colmn2_norm=colmn2/255;
colmn3_norm=colmn3/255;
colmn4_norm=colmn4/255;
colmn5_norm=colmn5/255;

data=[colmn1,colmn2_norm,colmn3_norm,colmn4_norm,colmn5_norm];

%%

input1=data(1:50,2:5);
input2=data(51:100,2:5);
input3=data(101:150,2:5);

input1_train=input1(1:20,:);
input2_train=input2(1:20,:);
input3_train=input3(1:20,:);
input_train=[input1_train ; input2_train ; input3_train];


 
input1_val=input1(21:35,:);
input2_val=input2(21:35,:);
input3_val=input3(21:35,:);
input_val=[input1_val ; input2_val ; input3_val];

input1_test=input1(36:50,:);
input2_test=input2(36:50,:);
input3_test=input3(36:50,:);
input_test =[input1_test ; input2_test;input3_test];

input_T1=[input_train;input_val];
input_T2=[input_train;input_test];
input_val=input_val;
input_test=input_test;

output1=colmn1(1:50,:);
output2=colmn1(51:100,:);
output3=colmn1(101:150,:);

output1_train=output1(1:20,:);
output2_train=output2(1:20,:);
output3_train=output3(1:20,:);
output_train =[output1_train ; output2_train;output3_train];

output1_val=output1(21:35,:);
output2_val=output2(21:35,:);
output3_val=output3(21:35,:);
output_val =[output1_val ; output2_val;output3_val];


output1_test=output1(36:50,:);
output2_test=output2(36:50,:);
output3_test=output3(36:50,:);
output_test =[output1_test ; output2_test;output3_test];

output_T1=[output_train;output_val];
output_T2=[output_train;output_test];
output_val=output_val;
output_test=output_test;
%%
for i=1:2
    if i == 1
        
        train_input=input_T1;
        train_output=output_T1;
        test_in= input_test ;
        test_out=output_test;
    end
    if i == 2
        
        train_input=input_T2;
        train_output=output_T2;
        test_in= input_val ;
        test_out=output_val;
    end
    
    rng(1)
    template = templateSVM(...
        'KernelFunction','polynomial', ...
        'PolynomialOrder',3, ...
        'KernelScale','auto', ...
        'BoxConstraint',1, ...
        'Standardize',true);
    
    model = fitcecoc(...
        train_input, ...
        train_output, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames',{'1','2','3'});
   
    predLabels = predict(model,test_in);
    predLabels = str2double(predLabels);
    correctPredictions=[length(predLabels)];
    for i=1:length(predLabels)
        if predLabels(i)==test_out(i);
            correctPredictions(i)=1;
        else
            correctPredictions(i)=0;
        end
    end
    
    testAccuracy = sum(correctPredictions)/length(test_out)*100
    [confmatCVDFT,grouporder] = confusionmat(test_out,predLabels);
    figure
    cm = confusionchart(test_out,predLabels);
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    cm.Title = 'Flower Irıs Confusion Matrix';
    dftTable = helperPrecisionRecall(confmatCVDFT);
    disp(dftTable)
end
