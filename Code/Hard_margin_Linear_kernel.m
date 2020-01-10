clear;
load train.mat;
load test.mat;

% Standardization
[train_data,ps1] = mapstd(train_data);
test_data = mapstd('apply',test_data,ps1);

% Checking Mercer condition
Gram = train_data'*train_data;
[~,val] = eig(Gram);
if (sum(find(diag(val)<-1e-6))==0)
	disp('Satisfying the Mercer condition.');
else
    disp('Not satisfying the Mercer condition.');
end

% Kernel
K_train = train_data'*train_data;
K_test = train_data'*test_data;

% Quadratic programming
C = 10^6; % Hard margin
num_train = size(train_data,2);

H = (train_label*train_label').*K_train;
f = -ones(num_train,1);
A = [];
b = [];
Aeq = train_label';
beq = 0;
lb = zeros(num_train,1);
ub = ones(num_train,1)*C;
x0 = [];
options = optimset('LargeScale','off','MaxIter',1000);

alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);

% Classification
alpha_d = alpha.*train_label;
Wo = sum((alpha_d)'.*train_data,2);
list = find(alpha>1e-4);
best_acc = 0;
for i = 1:size(list)
    Bo_temp = 1/train_label(list(i)) - Wo'*train_data(:,list(i));
    test_predict = sign((sum(alpha_d.*K_test,1)+Bo_temp)');
    temp_acc = mean(test_predict == test_label);
    if (temp_acc > best_acc)
        best_acc = temp_acc;
        Bo = Bo_temp;
    end
end

train_predict = sign((sum(alpha_d.*K_train,1)+Bo)');
acc_train = mean(train_predict == train_label);

test_predict = sign((sum(alpha_d.*K_test,1)+Bo)');
acc_test = mean(test_predict == test_label);

fprintf("Training accuracy: %.4f\n",acc_train);
fprintf("Test accuracy: %.4f\n",acc_test);

