clear;

labelcaps = {'Rock','Rap','Pop','Hiphop','Dance','Country','Jazz'};

fid = fopen('features.txt');
b = [];
tline = fgetl(fid);
%Input reading
while ischar(tline)
    %disp(tline);    
	ind = find(tline == ',');
	ind = [0 ind size(tline,2)+1];
	a =[];
	for i=1:size(ind, 2)-1
		a = cat(2, a, sscanf(tline(ind(i)+1:ind(i+1)-1), '%f'));
	end
	b = cat(1, b, a);
	tline = fgetl(fid);
end
fclose(fid);
clear a;
a = [] ; fold = []; Y = [];
pred = [];
[rows,cols]=size(b);
%creating test data
testd = [];
testr = 1;
for i=1:rows
    if mod(i,10) == 1
        testd(testr,:) = b(i,:);
        testr = testr + 1;
    end
end
testr = testr - 1;
for i=rows:-1:1
    if mod(i,10) == 1
        b(i,:) = [];
    end
end

label = b(:,cols:cols);
b(:,cols:cols)=[];

labelt = testd(:,cols:cols);
testd(:,cols:cols) = [];
[trows,tcols]=size(testd);

[rows,cols]=size(b);
display('Input reading is now finished');

% do the training and report training error
k=7;
[W, y_hat, labelstats, trainingerror] = multiclass_lda(b, label, k);

% test the performance on unseen data
n = numel(labelt);
y_hat = zeros(n,1);
X = [ones(n,1) testd];
%X = [testd];
%X = normalize(X);
for i=1:n
    [temp, y_hat(i)] = max(W * X(i,:)');
end

testerror = sum(y_hat~=labelt)/n

% report class distributions and label captions for the training set
labelstats
labelcaps

confmatrix = zeros(k,k);
for i=1:n
    confmatrix(labelt(i), y_hat(i)) = confmatrix(labelt(i), y_hat(i)) + 1;
end
confmatrix = confmatrix./sum(sum(confmatrix))

feature = 1;
for i=2:n
   if norm(W(:,i),2) > norm(W(:,feature),2)
       feature = i;
   end
end
feature

for i=1:k
   [temp,feature] =  max(abs(W(i,:)));
   [i feature]
end
