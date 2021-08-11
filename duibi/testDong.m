function rgb=testDong(DIR,path,xlspath,zu)

sprintf('当前第%d组，path:%s',zu,DIR)
% imgpwer=imread([path,'帧平均power480-640.jpg']);
if zu<5
    imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
else
    imgFiles = dir([DIR,'*.bmp']);
end
% [N, ~]= size(imgFiles);
N=200;
%%
tic

Img=imread([DIR ,imgFiles(1).name]);
img=im2double(Img);
% figure,imshow(Img),title('原低照图')

% A=zeros(size(img));
for k=2:N
    A=im2double(imread([DIR ,imgFiles(k).name]));
    img=img+A;
%     sprintf('正在读取%s',imgFiles(k).name)
end
img=img./N;

imout = Dong(img,0.8);

time=toc

figure(1),imshowpair(img,imout,'montage')
% figure(1);imshow(L);title('Input');
% figure(2);imshow(I);title('LIME');
%%
rgb=uint8(imout*255);
imwrite(rgb,[path,'帧平均Dong480-640','.jpg']);

gray = rgb2gray(rgb);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

quality=testquality(rgb,1);

value=[quality time];
value=roundn(value,-4);  %保留几位小数 b=vpa(value,4)保留有效数字

mRowRange='R7:W7';
writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\学习\小论文\data.xlsx',value,1,mRowRange);

