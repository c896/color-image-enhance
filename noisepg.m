clear
close all
clc

% path ='D:\Cammera\mode\modenoise6\';
path ='D:\ѧϰ\2021���Ʊ���\HSV��ε��\noise2\';
[status, msg, msgID] = mkdir(path)
% % imageName='House256.jpg';
% imageName='Lena512.png';
% % imageName='Baboon512.png';
% % img=imread(strcat([path,imageName]));

[imageName, pathname] = uigetfile('*.*','������ͼ��1');
% img=rgb2gray(imread([pathname,imageName]));
% img=double(img);
img=im2double(imread([pathname,imageName]));

% figure,imshow(img);

% I=imread('D:\Program Files\Polyspace\TU\���ն�\images\office\office_4.jpg');%������
% rect=[151 71 639 479]; %[x��� y��� �� ��] 640*480   0.25
% img=imcrop(I,rect);
% C= imresize(I,0.5);
% subplot(121),imshow(I);
% rectangle('Position',rect,'LineWidth',2,'EdgeColor','r')%��ʾͼ���������
% subplot(122),imshow(img);

% imwrite(img,['D:\Cammera\mode\result6\','6.jpg']);
samplenum = 300;  %��������

%%
% randn('seed',0); %ָ��seed�󣬵�һ�ε���rand()�õ��Ľ���ǡ�ȷ���ġ����൱�ڸ�rand�趨��һ��startpoint����ͬ��seed����Ӧ��startpoint��ͬ
%RNG:https://ww2.mathworks.cn/help/matlab/ref/rng.html

% sigma = 25;  % Noise standard deviation. it should be in the same
% S = 255;
% I_MAX = 255/S;
% sigma = sigma/S;
% z = img + sigma*randn(size(img));
%%
% y_noise_g = imnoise(img,'gaussian');%0.05 (default) | numeric scalar 
% imshow(y_noise_g,[]);
% y_noise_pg = imnoise(y_noise_g,'poisson');
% figure,imshow(y_noise_pg,[]);
% title('����-��˹noised_image');

%%
% randn('seed',0);    % fixes seed of random noise
% rand('seed',0);
% 
% disp('imnoising started')
% for i = 1:1:samplenum
%     %     img_noise=imnoise(img,'gaussian',0,0.02);
%     % img=imread(strcat([path,imageName]));
%     % img=im2double(img);
%     a=0.1^2;
%     b=0.02^2;
%     clipping_below=1;
%     clipping_above=1;
%     
%     if a == 0
%         z=img;
%     else
%         chi = 1/a;
%         z=poissrnd(max(0,chi*img))/chi;
%         %   imshow(z)
%     end
%     g=sqrt(b)*randn(size(img));
%     %   imshow(g)
%     z=z+g;
%     
%     if clipping_above
%         z=min(z,1);
%     end
%     if clipping_below
%         z=max(0,z);
%     end   
%     J=imdivide(z,10);
%     %     K=immultiply(J,10);
%     %     figure,imshow(z,[]);
%     %     title('����-��˹noised_image');
%     
%     %     figure,
%     %     subplot(121),imshow(J)
%     %     subplot(122),imshow(K)
%     % max(max(z))
%     nm=num2str(i);
%     if length(nm)==1
%         nm=['00' nm];
%     elseif length(nm)==2
%         nm=['0' nm];
%     end
% %     imwrite(J,[path,'poisson-gaussian-',nm,'.jpg']);
%     fprintf('����%s\n',strcat('noise-',nm,'.jpg'));% ��ʾ���ڴ����ͼ����
%     
% end
% mean2(J)
% % I_MAX=double(max(max(max(img))));
% % I_MAX=255;
% % PSNR = 10*log10(I_MAX^2/mean((img_noise(:)-img(:)).^2))

%% ��
img=double(img);
% peaks = [1 2 5 10 20 30 60 120];
peaks = [120]; % target peak values for the scaled image����ͼ���Ŀ���ֵ
sigmas = peaks/10;                % standard deviation of the Gaussian noise
reps = 10;
pp=1;

% Poisson scaling factor���ɱ�������
alpha = 1;

% mixed Poisson-Gaussian noise parameters:
peak = peaks(pp); % target peak value for the scaled image

% Gaussian component N(g,sigma^2)
sigma = sigmas(pp);
g = 0.0;

randn('seed',0);    % fixes seed of random noise
rand('seed',0);

for i = 1:1:samplenum
    
    nm=num2str(i);
    if length(nm)==1
        nm=['00' nm];
    elseif length(nm)==2
        nm=['0' nm];
    end
    
    scaling = peak/max(img(:));
    y = img*scaling;
    
    % z1 = alpha*poissrnd(y);
    % imshow(z1)
    % z2 = sigma*randn(size(y));
    % imshow(z2)
    Z = alpha*poissrnd(y) + sigma*randn(size(y)) + g;   
    Z=uint8(round(Z));
    imwrite(Z,[path,'poisson-gaussian-',nm,'.jpg']);
    fprintf('����%s\n',strcat('noise-',nm,'.jpg'));% ��ʾ���ڴ����ͼ����
end

figure,imshow(Z);
mean2(Z)

% max(max(y))
% figure,imshow(uint8(y),[]);


