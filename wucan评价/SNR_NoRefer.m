function SNR_No_Refer=SNR_NoRefer(img)
% clc
% clear
% [filename,pathname]=uigetfile('*.*','ͼ��');
% img=imread([pathname,filename]);

if (ndims(img)==3) % ��Ϊrgbͼ����ת�Ҷ�
   I_gray=rgb2gray(img);
else
   I_gray=img;
end
img=im2double(I_gray);
% if (max(img(:))<2)
%     img = img*255;
% end
[r,c,h] = size(img);

% Mean value
img_mean = mean(mean(img));

% Variance
img_var = sqrt(sum(sum((img - img_mean).^2)) / (r * c*h));
% img_var = std2(img);
SNR_No_Refer=img_mean/img_var;


%%S/N=1/C=img_var/Variance


