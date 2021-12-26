A=normrnd(-2.5,2,350,1);
B=normrnd(2.5,1,250,1);
C=[A;B];
select=randperm(600,420);
train=C(select);
test=setdiff(C,train);
self_mvnrnd(mean(train),cov(train),350,0.5,mean(test),cov(test),250,0.5)