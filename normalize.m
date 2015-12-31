function X_new = normalize(X)

eps=1e-6;

X_new = zeros(size(X));

for i=1:size(X,2)
   if norm(X(:,i))>eps
       X_new(:,i) = X(:,i)./norm(X(:,i));
   end
end

end
