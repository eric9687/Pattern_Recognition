close all
clear
clc
xita1=3;
xita0=6;
x=normrnd(0,1,100,1) ;


sigma1=0.5;sigma2=2;

e1=normrnd(0,sigma1,100,1) ;
y1=xita1.*x+xita0+e1;

e2=normrnd(0,sigma2,100,1) ;
y2=xita1.*x+xita0+e2;
for i=1:3
    k_y1=polyfit(x,y1,i);
    Y1(:,i)=polyval(k_y1,x);
    k_y2=polyfit(x,y2,i);
    Y2(:,i)=polyval(k_y2,x);
end

%% 画回归图
figure
plot(x,y1,'o')
for i=1:3
    hold on
    plot(x,Y1(:,i),'*')
end
xlabel('x')
ylabel('y')
title('δ=0.5')
legend({'真实' '线性' '一元二次' '一元三次'})


figure
plot(x,y2,'o')
for i=1:3
    hold on
    plot(x,Y2(:,i),'*')
end
xlabel('x')
ylabel('y')
title('δ=2')
legend({'真实' '线性' '一元二次' '一元三次'})

%% 计算RSS
for i=1:3
    RSS1(i)=sum((Y1(:,i)-y1).^2);
    RSS2(i)=sum((Y2(:,i)-y2).^2);

end
RSS=[RSS1;RSS2];
RSS=double(vpa(RSS,4))
