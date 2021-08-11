clc
clear 
path='D:\Cammera\7.02\';
%% 
if exist('filename','var') ~= 1
    [filename, user_canceled] = imgetfile;
    if user_canceled, error('canceled'); end;
end
I = imread(filename);
figure('name','Original'), imshow(I);

rect=[10 450 2559 1919]; %[x起点 y起点 宽 高] 640*480   0.25
% rect=[980 1380 1279 959]; %[x起点 y起点 宽 高]

B=imcrop(I,rect);
C= imresize(B,0.25);
% C= imresize(I,0.5);
subplot(121),imshow(I);
rectangle('Position',rect,'LineWidth',2,'EdgeColor','r')%显示图像剪切区域
subplot(122),imshow(C);
sprintf('正在处理%s',filename)


imwrite(C,[path,'正常光照图-480-640.jpg']);

%%
%------------图片尺寸转换----------------%
% for j = 1:N
%     A=imread([DIR ,imgFiles(j).name]);
%     f = 512 / max(size(A));
%     I = imresize(A,f);
% %     I= imresize(A,0.2);
%     %         I=power(im2double(B),0.4);
%     %         subplot(121),imshow(A);
%     %         subplot(122),imshow(I);
%     sprintf('正在处理%s',imgFiles(j).name)
%     imwrite(I,[path,imgFiles(j).name]);
%     
% end

%%
%------------图片旋转-----------------%
% for j = 1:N
%         [A,map]=imread([DIR ,imgFiles(j).name]);
%         B= imresize(A,0.2);
%         C=imrotate(B,-90);
% %         I=power(im2double(C),0.4);
% %         subplot(121),imshow(A);
% %         subplot(122),imshow(I);
%         sprintf('正在处理%s',imgFiles(j).name)
%         imwrite(C,[path,imgFiles(j).name]);
% end

toc