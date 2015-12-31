function y = oneofk(labels, k)

n = numel(labels);

y = zeros(n,k);

for i=1:n
    y(i,labels(i))=1;
end

end
