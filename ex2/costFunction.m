function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%thetaX = theta'.*X;
%HofTheta = zeros((size(X)),1);
%for i = 1:length(theta)
%  HofTheta = sigmoid(thetaX(:,i));
%end  

thetaX = X*theta;
HofTheta = sigmoid(thetaX);
product = -y.*log(HofTheta).-(1.-y).*log(1.-HofTheta);
J = sum(product)/m;

sumationOfGrad = sum((HofTheta .- y).*X);
grad = sumationOfGrad/m;





% =============================================================

end
