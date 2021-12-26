close all
clear
clc

%% 2
N=[10 100 1000];
for j=1:length(N)
    n=N(j);
    figure
    for i=1:3
       data=normrnd(0,1,n,1); 
%         clc;clear
%         x=randn(1,1000);
        %hist(x)
        [mu,sigma] = normfit(data);
        d=pdf('norm',data,mu,sigma);
        plot(data,d,'.')
%        plot(x,y);
       hold on
    end
    xx = -10:0.01:10;
    yy = normpdf(xx, 0, 1);
    plot(xx,yy);
    legend({'第1次' '第2次' '第3次' '标准'})
end

%% 3
x = [0:99];
y = [1:100];
data=unifrnd(x,y)./100;
[mu,sigma] = normfit(data);
d=pdf('norm',data,mu,sigma);
figure
plot(data,d,'.')
hold on
plot(data,ones(1,length(data)),'-')
legend({'正态分布模型' '标准均匀分布'})