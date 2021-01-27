function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%HofTheta = [zeros(47,1)];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
theta = [theta;zeros(min(size(X)) - length(theta),1)];
thetaX = theta'.*X;
HofTheta = sum(thetaX')';
numSqr = (HofTheta - y).^2;
numerator = sum(numSqr);
denominator = 2*m;
J = numerator/denominator;


% =========================================================================

end
