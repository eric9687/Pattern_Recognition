close all
clear
clc


X=[];
num_pic=randperm(10,7)-1;
for i=1:length(num_pic)
    pic_id=num_pic(i);
    path=[num2str(pic_id)];
    matrix_name=dir([path '/' '*.png']);
    matrix_name={matrix_name.name};
    TEMP=[];
    for j=1:length(matrix_name)
        pic=imread([path '/' matrix_name{j}]);
        pic=double(rgb2gray(pic));
        temp=[];
        for k=1:size(pic,1)
            temp=[temp pic(k,:)];
        end
        TEMP=[TEMP;temp];
    end
    data{i}=TEMP;
    X=[X;TEMP];
end

X=mapstd(X);
x_num=size(X,1)/300;


X_0=[];
X_1=[];
y_0=[];
y_1=[];
for i=1:x_num
    index=(i-1)*300+1:300*i;
    num0=round(length(index)*0.75);
    X_0=[X_0;X(index(1:num0),:)];
    X_1=[X_1;X(index(num0+1:end),:)];
    y_0=[y_0;ones(num0,1)*i];
    y_1=[y_1;ones(300-num0,1)*i];
end
X_0=X_0';
X_1=X_1';
y_0=y_0';
y_1=y_1';

Y_0=zeros(x_num,size(y_0,2));
Y_1=zeros(x_num,size(y_1,2));
for i=1:size(y_0,2)
   Y_0(y_0(i),i)=1;
end
for i=1:size(y_1,2)
   Y_1(y_1(i),i)=1;
end


net = trainSoftmaxLayer(X_0,Y_0);
Y = net(X_1);
y_predict=[];
for i=1:size(Y,2)
    n=find(Y(:,i)==max(Y(:,i)));
    y_predict=[y_predict n];
end

index_true=find((y_predict-y_1)==0);

zhunquelv=length(index_true)/length(y_predict)





