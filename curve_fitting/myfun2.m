 function F = myfun2(x,xdata)

        F = x(1)*xdata(:,1)+x(2)*xdata(:,2)+x(3)*xdata(:,3)+x(4)*xdata(:,4) ...
            +x(5).*xdata(:,1).*xdata(:,2)+x(6).*xdata(:,1).*xdata(:,3)+x(7).*xdata(:,1).*xdata(:,4) ...
            +x(8).*xdata(:,2).*xdata(:,3)+x(9).*xdata(:,2).*xdata(:,4)+x(10).*xdata(:,3).*xdata(:,4)+x(11);
 end