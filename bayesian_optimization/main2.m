close all
clear
clc
%% 2
a=1;
matrix_h=[1 1/4 1/16];
x=-1:0.001:1;
matrix_P=zeros(3,length(x));
figure
for i=1:length(matrix_h)
    h=matrix_h(i);
    n=find(x<0);
    matrix_P(i,n)=0;
    n=find(x>=0 & x<=a);
    matrix_P(i,n)=1./a.*(1-exp(-x(n)./h));
    n=find(x>a);
    matrix_P(i,n)=1./a.*(exp(a./h)-1).*exp(-x(n)./h);
    hold on
    plot(x,matrix_P(i,:))
end
legend({'h=1' 'h=1/4' 'h=1/16'})

%% 4
h=-a/(log(1-0.01*a));
P=zeros(1,length(x));
n=find(x<0);
P(n)=0;
n=find(x>=0 & x<=a);
P(n)=1./a.*(1-exp(-x(n)./h));
n=find(x>a);
P(n)=1./a.*(exp(a./h)-1).*exp(-x(n)./h);
index(1)=find(x==0);
index(2)=find(abs((x-0.05))==min(abs((x-0.05))));
figure
plot(x(index(1):index(2)),P(index(1):index(2)))