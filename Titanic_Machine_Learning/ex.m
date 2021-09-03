% Initialization
clear ; close all; clc

% Reading Training Parameters

%% x1=Pclass ; x2=gender ; x3=age ; x4=SibSp ; x5=Parch ; x6=fare ; x7=embarked
%feature normalized in Excel

X=csvread('Xin.csv');
y=csvread('yin.csv');

a=size(X);
m=a(1);
n=a(2);

input_layer_size=7;
hidden_layer1_size=150;
hidden_layer2_size=150;

fprintf('Program paused. Press enter to continue.\n');
pause;

%Computing Initial Theta , removing symmetry bias

epsilon_init = 0.12;
initial_Theta1 = rand(hidden_layer1_size, 1 + input_layer_size) * 2 * epsilon_init - epsilon_init;
initial_Theta2 = rand(hidden_layer2_size, 1 + hidden_layer1_size) * 2 * epsilon_init - epsilon_init;
initial_Theta3 = rand(1, 1 + hidden_layer2_size) * 2 * epsilon_init - epsilon_init;

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)]; %unroll parameters

%computing cost


lambda = 0;

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer1_size, hidden_layer2_size, 1, X, y, lambda);


costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer1_size, hidden_layer2_size, 1, X, y, lambda);
options = optimset('MaxIter', 1000);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):((hidden_layer1_size * (input_layer_size + 1))+(hidden_layer2_size * (hidden_layer1_size + 1)))), ...
                 (hidden_layer2_size), (hidden_layer1_size + 1));
Theta3 = reshape(nn_params((1 + ((hidden_layer1_size * (input_layer_size + 1))+(hidden_layer2_size * (hidden_layer1_size + 1)))):end), ...
                 1, (hidden_layer2_size + 1));
;				 


fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%Xval=csvread('Xval.csv');

%Xtest=csvread('Xtest.csv');
%pred = predict(Theta1, Theta2, Theta3, Xtest);
%save('ytest.csv' , 'pred');
%fprintf('\nValidation Set Accuracy: %f\n', mean(double(pred == y)) * 100);


