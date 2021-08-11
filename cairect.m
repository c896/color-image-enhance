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

rect=[10 450 2559 1919]; %[x��� y��� �� ��] 640*480   0.25
% rect=[980 1380 1279 959]; %[x��� y��� �� ��]

B=imcrop(I,rect);
C= imresize(B,0.25);
% C= imresize(I,0.5);
subplot(121),imshow(I);
rectangle('Position',rect,'LineWidth',2,'EdgeColor','r')%��ʾͼ���������
subplot(122),imshow(C);
sprintf('���ڴ���%s',filename)


imwrite(C,[path,'��������ͼ-480-640.jpg']);

%%
%------------ͼƬ�ߴ�ת��----------------%
% for j = 1:N
%     A=imread([DIR ,imgFiles(j).name]);
%     f = 512 / max(size(A));
%     I = imresize(A,f);
% %     I= imresize(A,0.2);
%     %         I=power(im2double(B),0.4);
%     %         subplot(121),imshow(A);
%     %         subplot(122),imshow(I);
%     sprintf('���ڴ���%s',imgFiles(j).name)
%     imwrite(I,[path,imgFiles(j).name]);
%     
% end

%%
%------------ͼƬ��ת-----------------%
% for j = 1:N
%         [A,map]=imread([DIR ,imgFiles(j).name]);
%         B= imresize(A,0.2);
%         C=imrotate(B,-90);
% %         I=power(im2double(C),0.4);
% %         subplot(121),imshow(A);
% %         subplot(122),imshow(I);
%         sprintf('���ڴ���%s',imgFiles(j).name)
%         imwrite(C,[path,imgFiles(j).name]);
% end

toc