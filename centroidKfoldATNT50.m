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
 
     gpF = test(1,:);
     group = train(1,:);
     [m,  w] = hist(train(1,:),unique(train(1,:)));
    [o p] = hist(test(1,:),unique(test(1,:)));
    s1 = m(1);
    
     tester = test(2:end,:);
     tester = tester';
     %group = trainer(1,1:end);
     x = train';
     [uv,~,idx] = unique(x(:,1));
     nu = numel(uv);
     x_sum = zeros(nu,size(x,2));
     for ii = 1:nu
       x_sum(ii,:) = sum(x(idx==ii,:));
     end
     x_sum = x_sum(1:end,2:end);
     x_div = x_sum/s1;
     dist = pdist2(tester,x_div);
     [~, class] = min(dist,[],2);
     output = uv(class);
     output = output';
     %disp(output);
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
