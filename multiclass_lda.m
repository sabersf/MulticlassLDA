function [W, y_hat, labelstats, trainingerror] = multiclass_lda(traindata, labels, k)

[n,d] = size(traindata);

% build one-of-k representation of the input
n1 = numel(labels);
T = zeros(n1,k);
for i=1:n
    T(i,labels(i))=1;
end
%T = [ones(n,1) T];

% input vector
X = [ones(n,1) traindata];
%X = [traindata];
X_pinv = pinv(X);

% compute the closed form solution for the parameters
W = X_pinv * T;
W = W';

% predict labels using y_hat(x) = argmax_i w_i'*x
y_hat = zeros(n,1);
for i=1:n
    [temp, y_hat(i)] = max(W * X(i,:)');
end

trainingerror = sum(y_hat~=labels)/n

for i=1:k
    labelstats(i) = sum(labels==i)/n;
end

end
