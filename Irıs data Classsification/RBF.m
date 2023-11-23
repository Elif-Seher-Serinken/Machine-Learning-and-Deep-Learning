clc;
clear;
close all;
tic
% get data
load fisheriris.mat

classnames = unique(species);
for i=1:3
    class(strcmp(species, classnames{i})) = i;
end
class = class';


%inputs data
data = [class meas];

% shuffle data
[m, n] = size(data);
[~, idx] = sort(rand(m,1));

for c = 1:m
    data(c,1:3) = data(idx(c),1:3);
end

% splite Data 70/30
PD = 0.70;
trainn = data(1:round(PD*length(data)),:); 
test = data(round(PD*length(data)):end,:);

% train

colmn1_train=trainn(:,2);
colmn2_train=trainn(:,3);
colmn3_train=trainn(:,4);
colmn4_train=trainn(:,5);
%

colmn1_norm_train=colmn1_train/255;
colmn2_norm_train=colmn2_train/255;
colmn3_norm_train=colmn3_train/255;
colmn4_norm_train=colmn4_train/255;

%

x_train = [colmn1_norm_train,colmn2_norm_train,colmn3_norm_train,colmn4_norm_train];

x_train = transpose(x_train);
y_train = trainn(:,1);
a=[1 0 0];
b=[0 1 0];
c=[0 0 1];

outputTrain = zeros(size(y_train,1),3);

       for j=1:size(y_train,1)
           if y_train(j,1) == 1
               outputTrain(j,:) = a;
           end
           if y_train(j,1) == 2
               outputTrain(j,:) = b;
           end
           if y_train(j,1) == 3
               outputTrain(j,:) = c;
           end
       end
y_train = outputTrain'; %(3x135)


% build model and train

Goal=1e-15;
Spread=5;
MaxNeuron=100;
net = newrb(x_train,y_train,Goal,Spread,MaxNeuron);


net.divideFcn = 'dividerand';  
net.divideMode = 'sample'; 

% splite train data for val and train data
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 20/100;

net.trainFcn = 'traingd'; 
net.performFcn = 'mse'; 

net.plotFcns = {'plotperform','plotregression','plottrainstate'};

net.trainParam.showWindow=true;
net.trainParam.showCommandLine=false;
net.trainParam.show=1;
net.trainParam.lr=0.1;
net.trainParam.epochs=100;
net.trainParam.goal=1e-15;
net.trainParam.max_fail=400;

[net,tr] = train(net,x_train,y_train);


% test normalaze and splite 
colmn1_test=test(:,2);
colmn2_test=test(:,3);
colmn3_test=test(:,4);
colmn4_test=test(:,5);

colmn1_norm_test=colmn1_test/255;
colmn2_norm_test=colmn2_test/255;
colmn3_norm_test=colmn3_test/255;
colmn4_norm_test=colmn4_test/255;

x_test = [colmn1_norm_test,colmn2_norm_test,colmn3_norm_test,colmn4_norm_test];

x_test = transpose(x_test);
y_test = test(:,1);
outputTest = zeros(size(y_test,1),3);

       for j=1:size(y_test,1)
           if y_test(j,1) == 1
               outputTest(j,:) = a;
           end
           if y_test(j,1) == 2
               outputTest(j,:) = b;
           end
           if y_test(j,1) == 3
               outputTest(j,:) = c;
           end
       end
y_test = outputTest';

% test 
Y = net(x_test);

% evaluate
pred = round(Y);
acc_count = nnz(pred==y_test); 
acc = acc_count/length(y_test)/3;
[per con] = confusion(y_test,Y);
perTest = 100 * (1 - per);

disp("accuracu : " + acc);
x(i)=plotconfusion(Y,y_test);
saveas(x(i),'confusion_rbf.jpg');
plotroc(pred,y_test);
toc