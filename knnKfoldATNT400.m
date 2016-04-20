data = dlmread('C:\Users\Dhanashri\Documents\Sem II\Data Mining\Proj2\ATNT400\ATNTFaceImages400.txt');
X = data(2:end,:);
row = size(X,2);%400
%fold = 5;
[uniVal,~,index] = unique(data(1,:));  
 cnt_uniVal = numel(uniVal);     %40
 rowPerFold = row / cnt_uniVal;  %10
     testFold = floor(rowPerFold / fold);   %2
 A = data;
accuracy = [];
n = testFold;
l = 1;
count = 0;
cnt = 1;
correctCount=0;

for i = 1:fold
     testFold = floor(rowPerFold / fold);   %2
     fld(i) = testFold;
 end
abs = mod(rowPerFold,fold); 
for i = 1:abs
   fld(i) = fld(i) + 1;
end

% while (l <= fold)
%      if l==fold && mod(rowPerFold,fold)>0
%         n = n - 1;
%      end
for i = 1:fold
        n = fld(i);

        
        col = [];
        test = [];
        train = [];
        
     for k = cnt:n:row
        col = [col cnt:(cnt+(n-1))];
        %count = 1;
        cnt = cnt + rowPerFold;        
    end
        col(col > row) = [];
        dup = A;
        B = dup(:,col);
        test = [test B];
    
    count = count + n;
    cnt = count + 1;
    l = l + 1;
    
    dup(:,col) = [];
    train = [train dup];
    
    [m,  w] = hist(train(1,:),unique(train(1,:)));
    [o p] = hist(test(1,:),unique(test(1,:)));
    s1 = m(1);

    gpF = test(1,:);
    innerTrain = train(2:end,:);
    innerTrain = innerTrain';
    innerTest = test(2:end,:);
    innerTest = innerTest';
    group = train(1,:);
    dist = pdist2(innerTest, innerTrain, 'euclidean');
    %kval=5;
    [~,index] = sort(dist,2,'ascend');
    grp = group(index);
    ind = grp(:,1:kval);
    G = mode(ind,2);
    output = G';
    count1 = 0;
    r1 = gpF - output;
     sizeCol = size(r1,2);
    for cnt1 = 1:length(r1)
        if(r1(cnt1) ~= 0)
            count1 = count1 + 1;
        end
    end
    r2 = (1- (count1/sizeCol))*100; 
    disp(r2);
    correctCount = correctCount + nnz(r1);
    accuracy = [accuracy r2];
end
accur = (1 - (correctCount/row))*100;
acc = mean(accuracy);
disp 'Final Accuracy= '
disp(accur);