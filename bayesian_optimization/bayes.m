function [R1_x,R2_x,result]=bayes(x,pw1,pw2)
m=numel(x) ;%�õ�����ϸ����Ŀ
R1_x=zeros(1,m); %��Ű�����X��Ϊ����ϸ������ɵ�������ʧ
R2_x=zeros(1,m) ;%��Ű�����X��Ϊ�쳣ϸ������ɵ�������ʧ
result=zeros(1,m); %��űȽϽ��
%���������ʷֲ� 
e1=-2;
a1=0.5;
e2=2;
a2=2;
%���վ���ֵ��
r11=0;
r12=10;
r21=1;
r22=0;
%�����������ֵ
for i=1:m
R1_x(i)=r11*pw1*normpdf(x(i),e1,a1)/(pw1*normpdf(x(i),e1,a1)+pw2*normpdf(x(i),e2,a2))+r21*pw2*normpdf(x(i),e2,a2)/(pw1*normpdf(x(i),e1,a1)+pw2*normpdf(x(i),e2,a2));
R2_x(i)=r12*pw1*normpdf(x(i),e1,a1)/(pw1*normpdf(x(i),e1,a1)+pw2*normpdf(x(i),e2,a2))+r22*pw2*normpdf(x(i),e2,a2)/(pw1*normpdf(x(i),e1,a1)+pw2*normpdf(x(i),e2,a2));
end

for i=1:m
    if R2_x(i)>R1_x(i)
        result(i)=0;
    else
        result(i)=1;
    end
end
 a=[-5:0.05:5];%ȡ�����㻭ͼ
 n=numel(a);
 R1_plot=zeros(1,n);
 R2_plot=zeros(1,n);
 for j=1:n
     R1_plot(j)=r11*pw1*normpdf(a(j),e1,a1)/(pw1*normpdf(a(j),e1,a1)+pw2*normpdf(a(j),e2,a2))+r21*pw2*normpdf(a(j),e2,a2)/(pw1*normpdf(a(j),e1,a1)+pw2*normpdf(a(j),e2,a2));
     R2_plot(j)=r12*pw1*normpdf(a(j),e1,a1)/(pw1*normpdf(a(j),e1,a1)+pw2*normpdf(a(j),e2,a2))+r22*pw2*normpdf(a(j),e2,a2)/(pw1*normpdf(a(j),e1,a1)+pw2*normpdf(a(j),e2,a2));
 end
 
 figure(1)
 hold on
 plot(a,R1_plot,'b-',a,R2_plot,'g-')
 for k=1:m
     if result(k)==0
         plot(x(k),-0.1,'b^') %����ϸ����������
     else
         plot(x(k),-0.1,'go')% �쳣ϸ����Բ��ʾ
     end
 end