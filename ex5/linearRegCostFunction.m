function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



% HofTheta = sigmoid(thetaX);
% product = -y.*log(HofTheta).-(1.-y).*log(1.-HofTheta);
% lambdaTheta = (lambda/2)*sum(theta(2:end).^2);
% J = (sum(product)+lambdaTheta)/m;

thetaX = theta'.*X;
HofTheta = sum(thetaX(:,1:end),2);
numSqr = (HofTheta - y).^2;
numerator = sum(numSqr);
denominator = 2*m;

lambdaTheta = lambda*sum(theta(2:end).^2);

J = (numerator+lambdaTheta)/denominator;







% =========================================================================

gradient = sum((HofTheta - y).*X)';
for i = 1:length(gradient)
  if i > 1
    grad(i) = (gradient(i)+(lambda*theta(i)))/m;
  else
    grad(i) = gradient(1)/m;
  end
end

grad = grad(:);

end
