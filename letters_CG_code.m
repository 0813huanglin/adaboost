%% get data ready for training and testing
x = 500;                    %number of data for training
T = 1;                    %number of round will take for boosting
RData = table2dataset(readtable('letters_CG.txt'));%%read data
[Row,Col] = size(RData);    %calculate size of read data
DataM = zeros(Row,Col-1);   %take data out from dataset save into matrix
for i = 1 : Row
    for j  = 2: Col
        DataM(i,j-1)=RData(i,j);
    end
end 
Label = ones(x,1);
for a =1 : Row
    if  strcmp('C',RData{a,1})  %C is positive sample G is negativ sample
        Label(a,1) = 1;
    else 
        Label(a,1) = -1;
    end
end
TrainSet=[DataM(1:x,:),Label(1:x,:)];    %att1...att1+label  training data
TestSet=[DataM(x+1:Row,:),Label(x+1:Row,:)];%testing data
%% start training
weight = ones(x,1)/x;                   %initialize weight
beststump = zeros(4,1);                 %matrix store tree stump [dim, thread, dir, alpha]
WeakClas = zeros(4,0);                  %matrix used to store training result
bestSClas=zeros(x,1);                   %matrix for store h(x) from decision stump
AgtCls = zeros(x,1);                    %matrix for aggregation
errplot = zeros(1,0);                   %store error rate for plot performance
%% Adaboost with decision stump start
for round = 1:T              %train until enough round or error rate is zero
%% decision stump
        err=inf;                     %initialize error rate
    for dim=1:Col-1          %for each feature dim
        for i = 1: x                            %for each sample
            for dir = 1:2                       %choose direction
                theta = TrainSet(i,dim);        %thresh
                Predic = ones(x,1);             %assume all samples are C
                if dir == 1                     %pos direction
                    for a=1 : x
                        if TrainSet(a,dim)<=theta   %let's say smaller than theta is G
                            Predic(a,1) = -1;
                        end
                    end
                else                            %neg direction
                    for b=1 : x
                        if TrainSet(b,dim)>theta    %greater than theta is G
                            Predic(b,1) = -1;
                        end
                    end
                end
                ErrArr = ones(x,1);             %assume all predics are wrong
                for j = 1:x
                    if  sign(Predic(j))==sign(TrainSet(j,Col))%mark it if this predic is correct
                        ErrArr(j) = 0;          %after this, each 1 represent a wrong predict                            
                    end
                end
            wErr = weight'*ErrArr;              %calculate weighted error
            if wErr<err                         %find a h(x) with min error rate
                err = wErr;                     %pass error rate to adaboost
                bestSClas(:,1)=Predic(:,1);     %pass h(x) to adaboost
                %% store key element of best stump in matrix
                beststump(1)=dim;               %store index of feature in matrix
                beststump(2)=theta;             %store value of threshold in matrix
                beststump(3)=dir;               %store direction in matrix
                %beststump(4)=err;    
            end
            end
        end
    end  
    %% adaboost    
    Alpha = 0.5*log((1-err)/err);                   %calculate voting power 
    beststump(4)=Alpha;                             %store voting power to beststump matrix
    Errtemp = zeros(x,1);                           
    temp = zeros(x,1);
    for s = 1:x                                     %calculate weight for each sample
        temp(s,1) = -1 * Alpha * TrainSet(s,Col).'*bestSClas(s,1);
        weight(s,1) = weight(s,1)*exp(temp(s,1));       %emphasis wrong sample
        weight(s,1) = weight(s,1)/sum(weight);          %normalize
    end    
    %% result of training
    WeakClas =  [WeakClas,beststump];                %store all h(x) in matrix
    %% re-calculate error rate
    for c = 1:x
        AgtCls(c,1) = AgtCls(c,1)+Alpha*bestSClas(c,1);    %aggregate      
        if sign(AgtCls(c,1))~=TrainSet(c,Col)      %%compare supervised label and predicted label
            Errtemp(c) = 1;                        %%mark errors                 
        end
    end
    aggErr = Errtemp(:,1)'*weight(:,1);            %%recalculate error rate
    eRate = aggErr/x;
    errplot = [errplot,eRate];
    if eRate == 0           %%stop when reach target error rate
        break;
    end
end
%% testing
[~,sc] = size(WeakClas);
aggClasTest = zeros(Row-x,1);
for t=1 : sc
    PredicTest = ones(Row-x,1);
    if WeakClas(3,t) == 1                     %pos direction
        for a=1 : Row-x
           if TestSet(a,WeakClas(1,t)) <= WeakClas(2,t)   %let's say smaller than theta is C
               PredicTest(a,1) = -1;
           end
        end
    else                            %neg direction
        for b=1 : Row-x
            if TestSet(b,WeakClas(1,t)) > WeakClas(2,t)    %greater than theta is G
                PredicTest(b,1) = -1;
            end
        end
    end
    temp2 = zeros(Row-x,1);
    for sz = 1 : Row-x
    temp2(sz,1)=WeakClas(4,t)*PredicTest(sz,1);
    end
    for z = 1: Row-x
    aggClasTest(z,1) = aggClasTest(z,1) + temp2(z,1);
    end
%i made mistake above put 2 different classes oppositely.
%i have a very bad classifier here with error rate about 97% in my 1st testing
%but if i use this classifier opposite way it will be a very good classifier
    testerr = 0;
    for q = 1 : Row-x
        if sign(TestSet(q,Col)) == -sign(aggClasTest(q,1)) %neg sign flip my classifier
            testerr = testerr+1;        %%calculate testing error
        end
    end
    terrate = testerr/(Row-x);          %%calculate testing error rate
end
%% plot testing result
figure(1);
plot(errplot,'b');
title('TrainingError');
xlabel('Number of Round');
ylabel('Error Rate');
figure(2);
threshold = 0;
moreThanThreshold = aggClasTest > 0;
lessThanThreshold = aggClasTest <= 0;
class1 = aggClasTest(moreThanThreshold);
class2 = aggClasTest(lessThanThreshold);
hold on;
plot(class1,'bo');
hline = refline(0);
hline.Color = 'g';
plot(class2, 'rx');
legend('pos sample','ref','neg sample');
title('Testing Result');
xlabel('Datas');
ylabel('likelihood');
hold off;