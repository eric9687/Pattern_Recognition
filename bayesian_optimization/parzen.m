function p=parzen(xi,x,h1,f)
%xiΪ������xΪ�����ܶȺ������Ա�����ȡֵ��
%h1Ϊ������Ϊ1ʱ�Ĵ���fΪ���������
%����x��Ӧ�ĸ����ܶȺ���ֵ
if isempty(f)
      %��û��ָ���������ͣ���ʹ����̬������
      f=@(u)(1/sqrt(2*pi))*exp(-0.5*u.^2);
end;
N=size(xi,2);
hn=h1/sqrt(N);
[X Xi]=meshgrid(x,xi);
p=sum(f((X-Xi)/hn)/hn)/N;