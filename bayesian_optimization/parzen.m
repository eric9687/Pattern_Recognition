function p=parzen(xi,x,h1,f)
%xi为样本，x为概率密度函数的自变量的取值，
%h1为样本数为1时的窗宽，f为窗函数句柄
%返回x对应的概率密度函数值
if isempty(f)
      %若没有指定窗的类型，就使用正态窗函数
      f=@(u)(1/sqrt(2*pi))*exp(-0.5*u.^2);
end;
N=size(xi,2);
hn=h1/sqrt(N);
[X Xi]=meshgrid(x,xi);
p=sum(f((X-Xi)/hn)/hn)/N;