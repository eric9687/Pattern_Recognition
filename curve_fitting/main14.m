close all
clear
clc
A=importdata('prostate_train.txt');
A=A.data;
X0=A(:,1:end-1);
y0=A(:,end);

A=importdata('prostate_test.txt');
A=A.data;
X1=A(:,1:end-1);
y1=A(:,end);

xdata = X0;           % example xdata
ydata = y0; % example ydata     

%% 没有考虑交叉项
x1 = lsqcurvefit(@(x,xdata) myfun(x,xdata),[1;1;1;1;1],xdata,ydata);

y_test_1 = x1(1)*X1(:,1)+x1(2)*X1(:,2)+x1(3)*X1(:,3)+x1(4)*X1(:,4)+x1(5);
RSS1=sum((y1-y_test_1).^2)
figure
plot(y_test_1,'r-')
hold on
plot(y1,'b--')
legend({'预测' '真实'})

%% 考虑交叉项
x2 = lsqcurvefit(@(x,xdata) myfun2(x,xdata),ones(1,11),xdata,ydata);
y_test_2 = x2(1)*X1(:,1)+x2(2)*X1(:,2)+x2(3)*X1(:,3)+x2(4)*X1(:,4) ...
            +x2(5).*X1(:,1).*X1(:,2)+x2(6).*X1(:,1).*X1(:,3)+x2(7).*X1(:,1).*X1(:,4) ...
            +x2(8).*X1(:,2).*X1(:,3)+x2(9).*X1(:,2).*X1(:,4)+x2(10).*X1(:,3).*X1(:,4)+x2(11);
RSS2=sum((y1-y_test_2).^2)
figure
plot(y_test_2,'r-')
hold on
plot(y1,'b--')
legend({'预测' '真实'})