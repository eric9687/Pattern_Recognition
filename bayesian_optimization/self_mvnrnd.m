function self_mvnrnd(varargin)
%���Զ�������ĺ���
if(nargin==8)
    %�ж���������Ƿ�Ϊ8
    w1=mvnrnd(varargin{1},varargin{2},varargin{3});
    %��һ��
    w2=mvnrnd(varargin{5},varargin{6},varargin{7});
    %�ڶ���
    figure(1);
    plot(w1(:,1),'bo');
    %��ɫoΪ��һ��
    hold on
    plot(w2(:,1),'g*');
    %��ɫ*Ϊ�ڶ���
    title('200�������������ɫoΪ��һ�࣬��ɫ*Ϊ�ڶ���');
    w=[w1;w2];
    n1=0;
    %��һ����ȷ���� 
    n2=0;
    %�ڶ�����ȷ���� 
    figure(2);
    %��Ҷ˹������
    for i=1:(varargin{3}+varargin{7})
        x=w(i,1);
        g1=mvnpdf(x,varargin{1},varargin{2})*varargin{4};
        g2=mvnpdf(x,varargin{5},varargin{6})*varargin{8};
        if g1>g2
            if 1<=i&&i<=varargin{3}
                n1=n1+1;
                %��һ����ȷ���� plot(x,y,'bo');
                %��ɫo��ʾ��ȷ��Ϊ��һ�������
                hold on;
            else
                plot(x,'r^');
                %��ɫ���������α�ʾ��һ������Ϊ�ڶ���
                hold on;
            end
        else if varargin{3}<=i&&i<=(varargin{3}+varargin{7})
                n2=n2+1;
                %�ڶ�����ȷ���� plot(x,y,'g*');
                %��ɫ*��ʾ��ȷ��Ϊ�ڶ��������
                hold on;
            else
                plot(x,'rv');
                %��ɫ���������α�ʾ�ڶ�������Ϊ��һ��
                hold on;
            end
        end
    end
    r1_rate=n1/varargin{3};
    %��һ����ȷ��
    r2_rate=n2/varargin{7};
    %�ڶ�����ȷ��
    gtext(['��һ����ȷ�ʣ�',num2str(r1_rate*100),'%']);
    gtext(['�ڶ�����ȷ�ʣ�',num2str(r2_rate*100),'%']);
    title('��С�����ʱ�Ҷ˹������');
else
    disp('ֻ�������������Ϊ8');
end
