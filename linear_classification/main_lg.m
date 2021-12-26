close all
clear
clc
fid_2=fopen('a.txt','wt');   %新建一个txt文件
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
X=mapminmax(A(:,1:end-1)')';
label=A(:,end);

index=randperm(length(label));
X_0=X(index(1:round(0.75*length(label))),:);
X_1=X(index(round(0.75*length(label))+1:end),:);
y_0=label(index(1:round(0.75*length(label))),:);
y_1=label(index(round(0.75*length(label))+1:end),:);

a =glmfit(X_0,y_0,'binomial', 'link', 'logit'); 
yy = round(glmval(a,X_1, 'logit')); 

delta=yy-y_1;
m=find(delta==0);

zhunquelv=length(m)/length(delta)