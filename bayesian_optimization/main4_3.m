A=normrnd(-2.5,2,350,1);
B=normrnd(2.5,1,250,1);
C=[A;B];
select=randperm(600,420);
train=C(select);
test=setdiff(C,train);
pw1=-2.5;
pw2=2.5;
x=train;
[R1_x,R2_x,result]=bayes(x,pw1,pw2)