data = dlmread('C:\Users\Dhanashri\Documents\Sem II\Data Mining\Proj2\ATNT400\ATNTFaceImages400.txt');
A = data;
X = data(2:end,:);
row = size(X,2);%400
%fold = 3;
fld=[];
[uniVal,~,index] = unique(data(1,:));  
cnt_uniVal = numel(uniVal);     %40
rowPerFold = row / cnt_uniVal;  %10
 for i = 1:fold
     testFold = floor(rowPerFold / fold);   %2
     fld(i) = testFold;
 end
abs = mod(rowPerFold,fold); 
accuracy = [];
l = 1;
count = 0;
cnt = 1;
correctCount = 0;
% 
  for i = 1:abs
         fld(i) = fld(i) + 1;
  end

%while (l <= fold)
for i = 1:fold
        n = fld(i);

%      if l==fold && mod(rowPerFold,fold)>0
%         n = n - 1;
%      end
%         
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
         disp(r2);
         correctCount = correctCount + nnz(r1);
         accuracy = [accuracy r2];
     
end
accur = (1 - (correctCount/row))*100;
 acc = mean(accuracy);
 disp 'Final Accuracy='
 %disp(acc);
 disp(accur);
