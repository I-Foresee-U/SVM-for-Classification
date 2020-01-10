load parameters.mat;
load train.mat;

% Standardization
[train_data,ps1] = mapstd(train_data);
eval_data = mapstd('apply',eval_data,ps1);

% Classification
K_eval = (train_data'*eval_data+1).^2;
eval_predicted = sign((sum((alpha.*train_label).*K_eval,1)+Bo)');
