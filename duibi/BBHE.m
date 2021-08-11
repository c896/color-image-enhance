% function Bi_HistogramEqualization()
% [f,p]=uigetfile('*.*','ѡ��ͼ���ļ�');
% if f
% I=imread(strcat(p,f));
% end
%
% Ir=I(:,:,1);%��ȡ��ɫ����
% Ig=I(:,:,2);%��ȡ��ɫ����
% Ib=I(:,:,3);%��ȡ��ɫ����
% I1=BBHE(Ir);
% I2=BBHE(Ig);
% I3=BBHE(Ib);
% In=cat(3,I1,I2,I3);  %cat���ڹ����ά����
% subplot(1,2,1);imshow(I);
% xlabel('A). ԭʼͼ��');
% subplot(1,2,2);imshow(mat2gray(In),[]);
% xlabel('B). ƽ������˫ֱ��ͼ���⻯');
% end

function A=BBHE(I)
Xm=floor(mean2(I));  %��ͼ��ҶȾ�ֵXm
Xmin=min(min(I));
Xmax=max(max(I));
[m,n]=size(I);
Xl=zeros(1,Xm+1);       %��¼ͼ���ڣ�Xmin,Xm)��Χ�ڵĻҶ�ֵ
Xu=zeros(1,256);       %��¼ͼ���ڣ�Xm,Xmax)��Χ�ڵĻҶ�ֵ
nl=0;
nu=0;
for i=1:m
    for j=1:n
        if I(i,j)<Xm || I(i,j)==Xm      %ͳ�ơ�ƽ��ֵ�ĸ����Ҷ�ֵ����������
            Xl(I(i,j)+1) = Xl(I(i,j)+1) + 1; %���ڻҶ�ֵΪ0��������������±겻��Ϊ0�����+1
            nl=nl+1;
        else                            %ͳ�ƣ�ƽ��ֵ�ĸ����Ҷ�ֵ����������
            Xu(I(i,j)+1) = Xu(I(i,j)+1) + 1;
            nu=nu+1;
        end
    end
end
X_(m+1)=Xm+1;
while(Xu(X_(m+1)+1))==0
    X_(m+1)=X_(m+1)+1;
end

Pl=Xl./nl;  %��¼��Ӧ�����Ҷ�ֵ�ĸ����ܶ�
Pu=Xu./nu;
Cl=Xl; %�ۼ��ܶȺ���
Cu=Xu;
Cl(1)=Pl(1);
Cu(X_(m+1))=Pu(X_(m+1));
for i=2:Xm+1
    Cl(i)=Pl(i) + Cl(i-1);
end
for i=X_(m+1)+1:256
    Cu(i)=Pu(i) + Cu(i-1);
end
%�Ҷ�ת������
fl=Cl;fu=Cu;
for i=1:Xm
    fl(i)= Xmin + Cl(i)*(Xm-Xmin);
end
for i=X_(m+1):256
    fu(i)= X_(m+1) + Cu(i)*(Xmax-X_(m+1));
end
%������ͼ��ϲ�
I_equal = I;
for i=1:m
    for j=1:n
        if I(i,j)<Xm || I(i,j)==Xm
            I_equal(i,j) = fl(I(i,j)+1);
        else
            I_equal(i,j) = fu(I(i,j)+1);
        end
    end
end
A=I_equal;
end