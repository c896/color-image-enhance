
%计算标准差 
%======================================================================
%标准差，反映了灰度相对于灰度均值的离散程度。标准差越大，则灰度级分布越分散，说明图像的质量越好
%======================================================================

clear all
[filename,pathname]=uigetfile('*.*','图像');
A=imread([pathname,filename]);
A=double(A);
Average=mean2(A(:))  %均值反映了图像的亮度，均值越大说明图像亮度越大，反之越小；
[M,N]=size(A);
sum=0;

for i=1:M
    for j=1:N
        sum=sum+(A(i,j)-Average)^2;
    end
end

SD=sqrt(sum/(M*N))