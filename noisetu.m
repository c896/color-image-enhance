clear
close all
clc

% path='C:\Users\CG\Desktop\5.28\��׼ͼ\';
path1 ='C:\Users\CG\Desktop\5.28\noise\';
% % imageName='House256.jpg';
% imageName='Lena512.png';
% % imageName='Baboon512.png';
% % img=imread(strcat([path,imageName]));

[imageName, path] = uigetfile('*.*','������ͼ��1');
img=im2double(imread([path, imageName]));

samplenum = 1;  %��������

%%
% randn('seed',0); %ָ��seed�󣬵�һ�ε���rand()�õ��Ľ���ǡ�ȷ���ġ����൱�ڸ�rand�趨��һ��startpoint����ͬ��seed����Ӧ��startpoint��ͬ
%RNG:https://ww2.mathworks.cn/help/matlab/ref/rng.html

% sigma = 25;  % Noise standard deviation. it should be in the same 
% S = 255;
% I_MAX = 255/S;
% sigma = sigma/S;
% z = img + sigma*randn(size(img));
%%
% y_noise_p = imnoise(img,'poisson');
% imshow(y_noise_p,[]); 
disp('imnoising started')
for i = 1:1:samplenum 
    y_noise_g = imnoise(img,'gaussian'); imshow(y_noise_g,[]); 
    y_noise_pg = imnoise(y_noise_g,'poisson');
    imwrite(y_noise_pg,[path1,'noise',num2str(i),imageName]);
    fprintf('����%s\n',strcat('noise',num2str(i),imageName));% ��ʾ���ڴ����ͼ���� 
end
figure,imshow(y_noise_pg,[]); 
title('����-��˹noised_image');

%%
% % img=imread(strcat([path,imageName]));
% % img=im2double(img);
% a=0.5^2;
% b=0.04^2;
% clipping_below=1;
% clipping_above=1;
% 
% if a == 0
%   z=img;
% else
%   chi = 1/a;
%   z=poissrnd(max(0,chi*img))/chi;
% %   imshow(z)
% end
% z=z+sqrt(b)*randn(size(img));
% 
% if clipping_above
%   z=min(z,1);
% end
% if clipping_below
%   z=max(0,z);
% end
% 
% figure,imshow(z,[]); 
% title('����-��˹noised_image');
% 
% J=imdivide(z,10);
% K=immultiply(K,10);
% 
% figure,
% subplot(121),imshow(J)
% subplot(122),imshow(K)
% % max(max(z))

%% ��
% % img=im2double(img);
% % peaks = [1 2 5 10 20 30 60 120]; 
% peaks = [2]; % target peak values for the scaled image����ͼ���Ŀ���ֵ
% sigmas = peaks/10;                % standard deviation of the Gaussian noise
% reps = 10;     
% pp=1;
% randn('seed',0);    % fixes seed of random noise
% rand('seed',0);
% 
% % mixed Poisson-Gaussian noise parameters:
% 
% peak = peaks(pp); % target peak value for the scaled image
% scaling = peak/max(img(:));
% y = img*scaling;
% 
% % Poisson scaling factor���ɱ�������
% alpha = 1;
% 
% % Gaussian component N(g,sigma^2)
% sigma = sigmas(pp);
% g = 0.0;
% % z1 = alpha*poissrnd(y); 
% % imshow(z1)
% % z2 = sigma*randn(size(y));  
% % imshow(z2)
% z = alpha*poissrnd(y) + sigma*randn(size(y)) + g;   
% 
% figure,imshow(z,[]); 

%%

disp('imnoising started')
for i = 1:1:samplenum 
    img_noise=imnoise(img,'gaussian',0,0.02); 
%     imwrite(img_noise,[path1,'noise',num2str(i),imageName]);
%     fprintf('����%s\n',strcat('noise',num2str(i),imageName));% ��ʾ���ڴ����ͼ���� 

end

figure(),
subplot(121),imshow(img)
subplot(122),imshow(img_noise)
% set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
% set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������

% I_MAX=double(max(max(max(img))));
I_MAX=255;
PSNR = 10*log10(I_MAX^2/mean((img_noise(:)-img(:)).^2))


