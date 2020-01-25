function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

disp('before')
disp(size(theta))
disp(size(grad))
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J = (1/(m)) * sum( ( (-y) .* log(sigmoid(X*theta)) ) - ( ( 1 - y) .* log(1 .- sigmoid( X*theta))) ) + ((lambda./(2.*m)) * (theta' .* [0; ones(size(theta, 1) - 1, 1)]') * theta);
disp('J')
size(J)


size(grad,1)
grad = (1/m) * (X' * (sigmoid(X*theta) - y)) + lambda / m * theta .* [0; ones(size(theta, 1) - 1, 1)];
% for i = 1:size(grad, 1)
%     if i==1;
%         grad(i,1) = 1./(m) .* ( (X(:,i))' *  ((sigmoid( X(:,i) * theta(i,1)) - y)) );
%     else 
%         grad(i,1) = (1./(m) .* ( (X(:,i))' *  ((sigmoid( X(:,i) * theta(i,1)) - y)) )) + ((lambda./m) .* theta(i,1));
%     end;
% end;
disp('after')
disp(size(grad))

% size((( X * theta)))





% =============================================================

end
