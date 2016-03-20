clear all; close all;
warning('off','all')

%% Read in Data
% Read training images
fid = fopen('train-images.idx3-ubyte','r','b');  
nummagic = fread(fid,1,'int32');    
num = fread(fid,1,'int32');  
row = fread(fid,1,'int32');   
col = fread(fid,1,'int32');   
for i=1:60000 %Take the first 20000 examples
    I_train{i} = reshape((double(fread(fid,[row col],'uchar')))',[784 1]);
end
fclose(fid);

% Read training labels
fid = fopen('train-labels.idx1-ubyte','r','b');  
nummagic = fread(fid,1,'int32');    
labnum = fread(fid,1,'int32');  

labels_train = double(fread(fid,60000,'uint8'));   
fclose(fid);

% Read test images
fid = fopen('t10k-images.idx3-ubyte','r','b');  
nummagic = fread(fid,1,'int32');    
num = fread(fid,1,'int32');  
row = fread(fid,1,'int32');   
col = fread(fid,1,'int32');   
for i=1:6000 %Take the first 2000 examples
    I_test{i} = reshape((double(fread(fid,[row col],'uchar')))',[784 1]);
end
fclose(fid);

% Read test labels
fid = fopen('t10k-labels.idx1-ubyte','r','b');  
nummagic = fread(fid,1,'int32');    
labnum = fread(fid,1,'int32');  

labels_test = double(fread(fid,6000,'uint8'));   
fclose(fid);


%% Pre processing Z score
X = cell2mat(I_train)'./256;
u = mean(X,2);
sigma = std(X,0,2);
X = (X - repmat(u,1,size(X,2)))./repmat(sigma,1,size(X,2));

X_test = cell2mat(I_test)';
X_test = [ones(size(I_test,2),1) X_test];

%% Mapping 0 to 10 to use during classification
labels_train(labels_train==0)=10; 
labels_test(labels_test==0)=10; 
num_labels = 10;
for i = 1:num_labels
   target(:,i) = labels_train == i;
end
%% Random weight initilization   
hidden_layer = 120; 

omega1 = randn(size(X,2)+1,hidden_layer)./sqrt(size(X,2)+1);
omega2 = randn(hidden_layer+1,num_labels)./sqrt(hidden_layer+1);
X = [ones(size(I_train,2),1) X];
batch = 500;
lambda = 0;
momentum = 0.9;
diff1 = 0;
diff2 = 0;
target_val = labels_train(1:10000);
X_val = X(1:10000,:);
for i = 1:56
    %% Split data into train and val  
    target_train = target(1:50000,:);
    X_train = X(1:50000,:);
    ix = randperm(size(X_train,1));
    X_train = X_train(ix,:);
    X_r = X_train(1:batch,:);
    %% Forward Propagation 
    a1 = X_r*omega1;
    g1 = ReLU(a1);
    g1 = [ones(batch,1) g1];
    a2 = g1*omega2;
    g2 = softmax(a2);
  
%% Back Propagation
    target_train = target_train(ix,:);
    target_r = target_train(1:batch,:);
    delta2 = target_r-g2;
    delta1 = (delta2*omega2').*sigmoid([ones(batch,1) a1]);
    % Regularization
    reg1 = lambda*[zeros(size(omega1, 1), 1) omega1(:, 2:end)];
    reg2 = lambda*[zeros(size(omega2, 1), 1) omega2(:, 2:end)];
    
    grad1 = delta1(:,2:end)'*X_r - reg1';
    grad2 = delta2'*g1 - reg2';
%% Checking the Gradient
% numgrad = checkGrad(omega1,omega2,target_train,X_train,hidden_layer,num_labels);
%% Gradient descent
alpha = 1e-4;
    omega2 = omega2 + alpha*grad2';
    diff2 = alpha*grad2' + momentum*diff2;
    omega1 = omega1 + alpha*grad1';
    diff1 = alpha*grad1 + momentum*diff1;
    %% Training Accuracy
    labels_train_acc = labels_train(ix);
    ta1 = X_train*omega1;
    tg1 = ReLU(ta1);
    tg1 = [ones(size(X_train,1),1) tg1];
    ta2 = tg1*omega2;
    tg2 = softmax(ta2);
%     error(i) = -sum(sum(target_train'*log(tg2),2));

    [~, gal] = max(tg2,[],2);
    att(i) = mean(gal == labels_train_acc(1:50000))*100;
    %% Checking validation
    v1 = ReLU(X_val*omega1);
    v1 = [ones(size(X_val,1),1) v1];
    v2 = softmax(v1*omega2);
%     errorv(i) = -sum(sum(target(50001:end,:)'*log(v2),2));
    
    [~, val] = max(v2,[],2);
    avv(i) = mean(val == target_val)*100;
    %% Testing accuracy 
    p1 = ReLU(X_test*omega1);
    p1 = [ones(size(X_test,1),1) p1];
    p2 = softmax(p1*omega2);
    [~, pred] = max(p2,[],2);
    acc(i) = mean(pred == labels_test)*100;
end









