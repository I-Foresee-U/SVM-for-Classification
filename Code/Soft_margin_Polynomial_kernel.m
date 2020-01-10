clear;
load train.mat;
load test.mat;

% Standardization
[train_data,ps1] = mapstd(train_data);
test_data = mapstd('apply',test_data,ps1);

p_list = [1,2,3,4,5];
C_list = [0.1,0.6,1.1,2.1];
acc_train = zeros(length(p_list),length(C_list));
acc_test = zeros(length(p_list),length(C_list));

for i = 1:length(p_list)
    p = p_list(i);
    fprintf("p = %d\n",p);
    
% Checking Mercer condition
    Gram = (train_data'*train_data+1).^p;
    [~,val] = eig(Gram);
    if (sum(find(diag(val)<-1e-6))~=0)
        disp('Not satisfying the Mercer condition.');
        continue
    else
        disp('Satisfying the Mercer condition.');   
    end
    
% Kernel
    K_train = (train_data'*train_data+1).^p;
    K_test = (train_data'*test_data+1).^p;
    
    for j = 1:length(C_list)
        C = C_list(j);
        fprintf("C = %.1f\n",C);
        
% Quadratic programming
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
        for k = 1:size(list)
            Bo_temp = 1/train_label(list(k)) - Wo'*train_data(:,list(k));
            test_predict = sign((sum(alpha_d.*K_test,1)+Bo_temp)');
            temp_acc = mean(test_predict == test_label);
            if (temp_acc > best_acc)
                best_acc = temp_acc;
                Bo = Bo_temp;
            end
        end
    
        train_predict = sign((sum(alpha_d.*K_train,1)+Bo)');
        acc_train(i,j) = mean(train_predict == train_label);

        test_predict = sign((sum(alpha_d.*K_test,1)+Bo)');
        acc_test(i,j) = mean(test_predict == test_label);

        fprintf("Training accuracy: %.4f\n",acc_train(i,j));
        fprintf("Test accuracy: %.4f\n",acc_test(i,j));
        
    end
end
