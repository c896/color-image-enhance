%计算均值、标准差、熵、平均梯度、相关系数、扭曲程度、偏差指数
%======================================================================
%均值，图像像素的灰度平均值，对人眼反映为平均亮度。
%标准差，反映了灰度相对于灰度均值的离散程度。标准差越大，则灰度级分布越分散。
%熵，图像的平均信息量。
%平均梯度，反映了图像的清晰程度，同时反映出图像中微小细节反差和纹理变换特征。
%相关系数，反映了两幅图像之间的相关性，相关系数越大两幅图像的相似程度越高。
%扭曲程度，反映影像的光谱失真程度。
%偏差指数，反映两幅图像在光谱信息上的匹配程度，偏差指数值越小，则说明融合后图
%像在提高了空间分辨率的同时，较好的保留了多光谱图像的光谱信息。
%=======================================================================
function quality=testquality(F,method)
% clc
% clear 
% [filename,pathname]=uigetfile('*.*','融合图像');
% F=imread([pathname,filename]);
% figure,imshow(F)
if ~exist('method','var') 
    method=0;
end
    
[~,~,n]=size(F);

% [filename,pathname]=uigetfile('*.*','多光谱图像');
% A=imread([pathname,filename]);

%[filename,pathname]=uigetfile('*.*','SAR图像');
%B=imread([pathname,filename]);
Fimage=double(F);

% A=double(A);
%B=double(B);

%求融合图像的灰度均值
mean=mean2(Fimage(:));
% mean=mean2(im2double(F));
% jb=Fimage(:);
%求标准偏差
std=std2(Fimage(:));
% std=std2(im2double(F));
%求熵
if n==3
    ent=entropy(F(:));%彩色熵
%     F_gray=rgb2gray(F);
else
%     F_gray=F;
    ent=entropy(F(:));%灰度熵
end
% ent_gray=entropy(F_gray(:));%灰度熵

[mf,nf,kf]=size(F);
q=0;

%求融合图像的平均梯度
% for k=1:1:kf
% grad=0;
% for i=1:1:mf-1
%     for j=1:1:nf-1
%         q=q+(sqrt(((Fimage(i,j,k)-Fimage(i+1,j,k))^2+(Fimage(i,j,k)-Fimage(i,j+1,k))^2)/2));
%     end
% end
% grad=grad+q/((mf-1)*(nf-1));
% end
% grad=grad./3;

for i=1:1:mf-1
    for j=1:1:nf-1
        q=q+(sqrt(((Fimage(i,j)-Fimage(i+1,j))^2+(Fimage(i,j)-Fimage(i,j+1))^2)/2));
    end
end
grad=q/((mf-1)*(nf-1));

%%
switch method
    case 0
        %求对比度
        cg=duibidu8(F);
        %求SNR_No_Refer 无参信噪比
        SNR_No_Refer=SNR_NoRefer(F);
        quality=[mean std ent grad cg SNR_No_Refer ];
        disp('     均值     标准差     信息熵    平均梯度  对比度  无参信噪比');
        disp(quality);%std ,ent ,grad ,c ,warp, bras); 
    case 1
        cg=duibidu8(F);
        brisqueI = brisque(F);
        niqeI = niqe(F);
%         piqeI = piqe(F);
        quality=[mean std ent grad cg brisqueI niqeI];
        disp('     均值     标准差     信息熵   平均梯度   对比度     BRISQUE    NIQE');
        disp(quality);%std ,ent ,grad ,c ,warp, bras);   
    case 2
        brisqueI = brisque(F);
        niqeI = niqe(F);
%         piqeI = piqe(F);
        quality=[mean std ent grad brisqueI niqeI piqeI];
        disp('     均值     标准差     信息熵   平均梯度    BRISQUE    NIQE');
        disp(quality);%std ,ent ,grad ,c ,warp, bras); 
end

%---------------------------%
%%
% %求相关系数 反映光谱保持性能
% rmul=imresize(A,[mf,nf],'bicubic');
% c=corr2(rmul(:),Fimage(:));
% 
% %求扭曲程度 直接反映影像的光谱失真程度
% q1=0;
% for i=1:1:mf
%     for j=1:1:nf
%         q1=q1+abs(Fimage(i,j)-rmul(i,j));
%     end
% end
% warp=q1/(mf*nf);
% 
% %求偏差指数
% q2=0;
% for i=1:1:mf
%     for j=1:1:nf
%         q2=q2+abs(Fimage(i,j)-rmul(i,j))/rmul(i,j);
%     end
% end
% bras=q2/(mf*nf);
% 
% result=zeros(1,7);
% result=[mean std ent grad c warp bras];
% disp('     均值     标准差       熵     平均梯度   相关系数   扭曲程度   偏差指数');
% disp(result );%std ,ent ,grad ,c ,warp, bras); 
% 
