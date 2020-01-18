function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% size(X)
% size(y)
% size(theta)
% disp(X * theta)
% disp((X * theta) .-  y)
% disp(((X * theta) .-  y) .^ 2)
% disp(sum(((X * theta) .-  y) .^ 2))
J = 1/(2*m) * (sum(((X * theta) .-  y) .^ 2)) ;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

end
