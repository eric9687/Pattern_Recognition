close all
clear
clc
fid_2=fopen('a.txt','wt');   
phns = ['breast-cancer-wisconsin.txt'];   %Ҫ��ȡ���ĵ����ڵ�·��
fid = fopen (phns);   %���ĵ�
while ~feof(fid)
    str = fgetl(fid);   % ��ȡһ��, str���ַ���
 n=strfind(str,'?');   
 if isempty(n)
   fprintf(fid_2,'%s\n',str);%�µ��ַ���д�뵱�½���txt�ĵ��� 
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
%���²���Ϊfisher�㷨��ʵ��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����������ֵ
m1=mean(w1)';
m2=mean(w2)';
%s1��s2�ֱ�����ʾ��һ�ࡢ�ڶ���������������ɢ�Ⱦ���
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
%������������ɢ�Ⱦ���Sw
Sw=s1+s2;
%����fisher׼����ȡ����ֵʱ�Ľ�w
w=inv(Sw)*(m1-m2);
%������ֵw0
ave_m1 = w'*m1;
ave_m2 = w'*m2;
w0 = (ave_m1+ave_m2)/2; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����Ϊ���Բ���
%����ginput���ѡȡ��Ļ�ϵĵ㣨������ȡ10���㣩
%����ɸ��ݵ��λ���Զ�����ʾ�������Ǹ���
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









