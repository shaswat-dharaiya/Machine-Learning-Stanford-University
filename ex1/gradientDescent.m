function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %  



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    

    thetaX = theta'.*X;
    HofTheta = thetaX(:,1) + thetaX(:,2);
    numSqr = (HofTheta - y);
    derivative0 = sum(numSqr.*X(:,1));
    derivative1 = sum(numSqr.*X(:,2));
    theta0 = theta(1) - alpha*derivative0/m;
    theta1 = theta(2) - alpha*derivative1/m;
    theta = [theta0;theta1];      

end

end
