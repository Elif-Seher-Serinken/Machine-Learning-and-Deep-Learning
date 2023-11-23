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
a=[1 0 0];
b=[0 1 0];
c=[0 0 1];

output = zeros(size(colmn1,1),3);

       for j=1:size(colmn1,1)
           if colmn1(j,1) == 1
               output(j,:) = a;
           end
           if colmn1(j,1) == 2
               output(j,:) = b;
           end
           if colmn1(j,1) == 3
               output(j,:) = c;
           end
       end
output = output;
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

input_T1=[input_train;input_val]';
input_T2=[input_train;input_test]';
input_val=input_val';
input_test=input_test';
output1=output(1:50,:);
output2=output(51:100,:);
output3=output(101:150,:);

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

output_T1=[output_train;output_val]';
output_T2=[output_train;output_test]';
output_val=output_val';
output_test=output_test';
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
    
    % build model and train
    n_hidden_nodes = 10;
    net = feedforwardnet(n_hidden_nodes);
       
    net.trainParam.epochs = 200; 
    net.trainParam.max_fail = 500;
    
    net.divideFcn = 'divideind';  
    net.divideMode = 'sample'; 
    
    % splite train and val for train model
    net.divideParam.trainInd= 1:60; % training data indices 
    net.divideParam.valInd= 61:105; % validation data indices
    
    
    net.trainFcn = 'traingd';  %gradient descent
    net.performFcn = 'mse'; %This argument defines the function used to measure the network’s performance
    
    net.plotFcns = {'plotperform','plotregression','plottrainstate'};
    
    net.trainParam.showWindow=true;
    net.trainParam.showCommandLine=false;
    net.trainParam.show=1; 
    net.trainParam.lr=0.3; %learning rate
    
    net.trainParam.goal=1e-12;
    
    
    net.layers{1}.transferFcn='tansig';
    net.layers{2}.transferFcn='purelin';
    % train the network
    %%
    [net,tr] = train(net,train_input,train_output);
    
    
    
    %%
    % test 
    Y = net(test_in);
    
    % evaluate
    pred = round(Y);
    
    [per con] = confusion(pred,test_out);
    perTest = 100 * (1 - per);
       
    fprintf('Fold = %d [Accuracy = %2.2f%%] \n',i, perTest);
    x(i)=plotconfusion(Y,test_out);
    if i==1
            saveas(x(i),'confusion_MLP_1.jpg');  
        end
        
     if i==2
          saveas(x(i),'confusion_MLP_2.jpg');  
     end
    plotroc(pred,test_out);
end