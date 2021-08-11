%-------------�����ȽϷ���ѡ��٤����ǿpower���ã����ۿ��Ÿ���ͣ����������Ӧֱ��ͼ���⻯adapthisteq--------%
clc
clear
path='C:\Users\CG\Desktop\5.28\';
% I = imread('tire.tif');
I = im2double(imread('C:\Users\CG\Desktop\5.28\IMG_3651.jpg'));
f = 512 / max(size(I));
S = imresize(I,f);
%------------------------RGBͨ��-------------------------%
%%
% img=power(S,0.4);
% img=real(img);
% figure,imshow(img)
% imwrite(img,[path,'rgb��ǿͼpower','.jpg']);

%%
% tic

% q=32;
% Ir=histeq(S(:, :, 1),q);
% Ig=histeq(S(:, :, 2),q);
% Ib=histeq(S(:, :, 3),q);

%%
% Ir=adapthisteq(S(:,:,1),'NumTiles',[2 2],'clipLimit',0.05,'Distribution','uniform');
% Ig=adapthisteq(S(:,:,2),'NumTiles',[2 2],'clipLimit',0.05,'Distribution','uniform');
% Ib=adapthisteq(S(:,:,3),'NumTiles',[2 2],'clipLimit',0.05,'Distribution','uniform');

%%
% J = cat(3, Ir,Ig,Ib);
% toc
% figure;
% imshow(J);
% imwrite(J,[path,'rgb��ǿͼhisteq','.jpg']);
% ssimval = ssim(J,S)

%%
% J1 = adapthisteq(I(:,:,1),'clipLimit',0.02,'Distribution','rayleigh');
% J2 = adapthisteq(I(:,:,2),'clipLimit',0.02,'Distribution','rayleigh');
% J3 = adapthisteq(I(:,:,3),'clipLimit',0.02,'Distribution','rayleigh');
% J=cat(3,J1,J2,J3);
% imshowpair(I,J,'montage');
% title('Original Image (left) and Contrast Enhanced Image (right)')

%%
%----------------------YUVͨ��-----------------%
% [width,height,m]=size(S);
% colorspace=' ';
% [I_yuv, img_max, img_min] = rgb_to(S,colorspace);
% I_yuv = reshape(I_yuv, [width, height, 3]);
%%
% Img=power(I_yuv,0.4);
% % imshow(I_yuv);
% I_est = rgb_to(Img, colorspace, true, img_max, img_min);
% I_est=reshape(I_est, [width,height, 3]);
% imshow(I_est)
% imshowpair(S,I_est,'montage');
% imwrite(I_est,[path,'yuv��ǿͼpower','.jpg']);
%%
%--------------�Աȶ����޵�����Ӧֱ��ͼ���⻯-----------------%
%'NumTiles',[8 8] ��ͼ��ֳ� 8 �� 8 �е�ͼ�顣 'uniform'ƽֱ̹��ͼ,'rayleigh'����ֱ��ͼ,'exponential'����ֱ��ͼ

% I1 = adapthisteq(I_yuv(:,:,1),'NumTiles',[2 2],'clipLimit',0.1,'Distribution','uniform');
% I2 = adapthisteq(I_yuv(:,:,2),'NumTiles',[2 2],'clipLimit',0.1,'Distribution','uniform');
% I3 = adapthisteq(I_yuv(:,:,3),'NumTiles',[2 2],'clipLimit',0.1,'Distribution','uniform');

%%
%-----------ֱ��ͼ���⻯---------------%
% % histeq(I,n) �任�Ҷ�ͼ�� I�����ؾ��� n ����ɢ�Ҷȼ��ĻҶ�ͼ�� J ��ӳ�䵽 J �� n ���Ҷȼ���ÿ����������ظ���������ȣ���� J ��ֱ��ͼ����ƽ̹
% q=32;
% I1=histeq(I_yuv(:,:,1),q);
% I2=histeq(I_yuv(:,:,2),q);
% I3=histeq(I_yuv(:,:,3),q);

%%
%--------������ǿ��ͼ���ֵӳ�䵽��ֵ���Զ�����������ǿ����ͺ���ߵ� 1%��Ĭ��ֵ�����ݽ��б��ʹ����Ӷ����ͼ��ĶԱȶ�----------

% I1=imadjust(I_yuv(:,:,1),[0 1],[]);
% I2=imadjust(I_yuv(:,:,2),[0 1],[]);
% I3=imadjust(I_yuv(:,:,3),[0 1],[]);

%%
% Img=cat(3,I1,I2,I3);
% I_est = rgb_to(Img, colorspace, true, img_max, img_min);
% I_est=reshape(I_est, [width,height, 3]);
% imshow(I_est)
% imshowpair(S,I_est,'montage');
% imwrite(I_est,[path,'yuv��ǿͼimadjust','.jpg']);

%%
%---------------����ͼ��tif-----------------%
% [X,map] = imread('shadow.tif'); %��ɫͼ map �� double ���͵� 256��3 ����
% shadow = ind2rgb(X,map); %������ͼ��ת��Ϊ RGB ͼ��
% % disp(['Range of RGB image is [',num2str(min(RGB(:))),', ',num2str(max(RGB(:))),'].'])
% %��� RGB ͼ���ֵ�Ƿ��� [0, 1] ��Χ�ڡ�
% shadow_lab = rgb2lab(shadow); %rgbתlab�ռ�
% max_luminosity = 100;
% L = shadow_lab(:,:,1)/max_luminosity;
% 
% shadow_imadjust = shadow_lab;
% shadow_imadjust(:,:,1) = imadjust(L)*max_luminosity;
% shadow_imadjust = lab2rgb(shadow_imadjust);
% 
% shadow_histeq = shadow_lab;
% shadow_histeq(:,:,1) = histeq(L)*max_luminosity;
% shadow_histeq = lab2rgb(shadow_histeq);
% 
% shadow_adapthisteq = shadow_lab;
% shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
% shadow_adapthisteq = lab2rgb(shadow_adapthisteq);

% figure
% montage({shadow,shadow_imadjust,shadow_histeq,shadow_adapthisteq},'Size',[1 4])
% title("Original Image and Enhanced Images using imadjust, histeq, and adapthisteq")