close all
clear
clc
fid_2=fopen('a.txt','wt');   
phns = ['breast-cancer-wisconsin.txt'];   %要读取的文档所在的路径
fid = fopen (phns);   %打开文档
while ~feof(fid)
    str = fgetl(fid);   % 读取一行, str是字符串
 n=strfind(str,'?');   
 if isempty(n)
   fprintf(fid_2,'%s\n',str);%新的字符串写入当新建的txt文档中 
 end
end 
 fclose(fid); 
 fclose(fid_2); 

A=importdata('a.txt');


X=mapstd(A(:,1:10));
label=A(:,11);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
index=randperm(length(label));
X_train=X(index(1:round(0.75*length(label))),:);
X_test=X(index(round(0.75*length(label))+1:end),:);
y_train=label(index(1:round(0.75*length(label))),:);
y_test=label(index(round(0.75*length(label))+1:end),:);
 
n1=find(y_train==0);
w1=X(n1,:);
n2=find(y_train==1);
w2=X(n2,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下部分为fisher算法的实现
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算样本均值
m1=mean(w1)';
m2=mean(w2)';
%s1、s2分别代表表示第一类、第二类样本的类内离散度矩阵
s1=0;
[row1,colum1]=size(w1);
for i=1:row1
     s1 = s1 + (w1(i,:)-m1)'*(w1(i,:)-m1);
end
s2=0;
[row2,colum2]=size(w2);
for i=1:row2
     s2 = s2 + (w2(i,:)- m2)' *(w2(i,:) - m2);
end
%计算总类内离散度矩阵Sw
Sw=s1+s2;
%计算fisher准则函数取极大值时的解w
w=inv(Sw)*(m1-m2);
%计算阈值w0
ave_m1 = w'*m1;
ave_m2 = w'*m2;
w0 = (ave_m1+ave_m2)/2; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%以下为测试部分
%利用ginput随机选取屏幕上的点（可连续取10个点）
%程序可根据点的位置自动地显示出属于那个类
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class=[];
for i=1:length(y_test)
  sample= X_test(i,:);

    if(sample*w - w0>0)
         class=[class; 0];
    else
         class=[class; 1];
    end
end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
er=class-y_test;
m=find(er==0);
zhunquelv=length(m)/length(er)









