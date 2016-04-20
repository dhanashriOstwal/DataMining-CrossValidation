load('C:\Users\Dhanashri\Documents\Sem II\Data Mining\Proj2\ATNT400\data645x50.mat');
A = data;
X = data(2:end,:);
row = size(X,2);%400
%fold = 5;
[uniVal,~,index] = unique(data(1,:));  
 cnt_uniVal = numel(uniVal);     %40
 rowPerFold = row / cnt_uniVal;  %10
 testFold = ceil(rowPerFold / fold);   %2

accuracy = [];
n = testFold;
l = 1;
count = 0;
cnt = 1;
while (l <= fold)
     if l==fold && mod(rowPerFold,fold)>0
        n = n - 1;
     end
        
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
    l = l + 1
    
    dup(:,col) = [];
    train = [train dup];
    
    [m,  w] = hist(train(1,:),unique(train(1,:)));
    [o, p] = hist(test(1,:),unique(test(1,:)));
    s1 = m(1);
    gpF = test(1,:);
    group = train(1,:);
    Xtest = test(2:end,:);
    Xtrain = train(2:end,:);
    %group = data(1,1:end);
    x = train';
    [uv,~,idx] = unique(x(:,1));
    nu = numel(uv);
    X = zeros(nu,size(x,1));
    [a,~]=hist(idx,unique(idx));
    start = 1;
    final = a;
    for rows = 1:nu
        for columns = start:final(rows)
            X(rows,columns)=1;
        end
        start = final(rows) + 1;
        final = final + a(rows);
    end

    Ytrain = X;

    Bx = (pinv(Xtrain')) *  double(Ytrain')  ; % (XX')^{-1} X  * Y'
    Ytrain1 = Bx' * Xtrain;
    Ytest1 = Bx' * Xtest;

    [Ytest2value,  Ytest2]= max(Ytest1,[],1);
    [Ytrain2value,  Ytrain2]= max(Ytrain1,[],1);
    output = Ytest2;
    count1 = 0;
    r1 = gpF - output;
    sizeCol = size(r1,2);
    for cnt1 = 1:length(r1)
        if(r1(cnt1) ~= 0)
            count1 = count1 + 1;
        end
    end
    r2 = (1- (count1/sizeCol))*100; 
    accuracy = [accuracy r2];
end
acc = mean(accuracy);
disp 'Final Accuracy='
disp(acc);
