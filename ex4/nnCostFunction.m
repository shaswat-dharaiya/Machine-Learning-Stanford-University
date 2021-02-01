function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Unrolling y - output matrix. ie. from 5000x1 to 5000x10
Y = [zeros(size(y,1),num_labels)];
for i =1:size(y,1)
  Y(i,y(i)) = 1;
end

% Input Layer with a(1)0 added.
a1 = [ones(size(X,1),1) X];

% Hidden Layer #1 with a(2)0 added.
z2 = a1*Theta1';
a2 = [ones(size(X,1),1) sigmoid(z2)];

% Output Layer
z3 = a2*Theta2';
a3 = sigmoid(z3);

HofTheta = a3;
product = -Y.*log(HofTheta).-(1.-Y).*log(1.-HofTheta);

lambdaTheta1 = 0;
lambdaTheta2 = 0;

lambdaTheta1 = sum(sum(Theta1(:,2:end).^2));
lambdaTheta2 = sum(sum(Theta2(:,2:end).^2));

J = (sum(sum(product))+ (lambda/2)*(lambdaTheta1 + lambdaTheta2))/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

for t =1:m

  % Forward Propagation
  % Step 1
  % Input Layer with a(1)0 added.
  a1_BackProp = [1 X(t,:)];

  % Hidden Layer #1 with a(2)0 added.
  z2_BackProp = a1_BackProp*Theta1';
  a2_BackProp = [1 sigmoid(z2_BackProp)];

  % Output Layer
  z3_BackProp = a2_BackProp*Theta2';
  a3_BackProp = sigmoid(z3_BackProp);

  % Backward Propagation
  % Step 2
  delta3_BackProp = a3_BackProp - Y(t,:);

  % Step 3
  delta2_BackProp = ((Theta2'*delta3_BackProp')(2:end).*sigmoidGradient(z2_BackProp'))';

  % Step 4
  Theta2_grad = Theta2_grad + delta3_BackProp'*a2_BackProp;
  Theta1_grad = Theta1_grad + delta2_BackProp'*a1_BackProp;
end
% Step 5
Theta1_grad(:,2:end) = (Theta1_grad(:,2:end)+(lambda*Theta1(:,2:end)));
Theta2_grad(:,2:end) = (Theta2_grad(:,2:end)+(lambda*Theta2(:,2:end)));
grad = [Theta1_grad(:) ; Theta2_grad(:)]/m;
end
