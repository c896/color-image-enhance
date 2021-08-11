function result1=zmretinex(DIR,path,xlspath,zu)
% % clc
% clear 
% close all
sprintf('��ǰ��%d�飬path:%s',zu,DIR)
% imgpwer=imread([path,'֡ƽ��power480-640.jpg']); 
if zu<5
    imgFiles = dir([DIR,'*.jpg']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
else
    imgFiles = dir([DIR,'*.bmp']);
end
% [N, ~]= size(imgFiles);
N=200;
tic

I=imread([DIR ,imgFiles(1).name]);
img=im2double(I);
% figure,imshow(img),title('ԭ����ͼ')

% A=zeros(size(img));
for k=2:N
    A=im2double(imread([DIR ,imgFiles(k).name]));
    img=img+A;
%     sprintf('���ڶ�ȡ%s',imgFiles(k).name)
end
img=img./N;

% % figure;imshow(I),title('ԭ���ն�ͼ֡ƽ��');
% imwrite(img,[path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% 
% % mean2(uint8(round(img*255))) % mean2(img)*255
% % img0=imread([path,'ԭ���ն�֡ƽ��480-640','.jpg']);
% % mean2(img0)
%%
%----------------retinex��ǿ------------------%
%     f = 512 / max(size(I));
%     I = imresize(I,f);
%     I(I < 0) = 0;I(I > 1) = 1;

% a=0.01;b=0.2;c=1; 
% icount=8;
a=0.008;b=0.3;c=1;
icount=4;
step=2.4;%decide the times  %step=2.2;

I=img+1;
Z=log2(I); %Z=log(I);
Z=Z/max(Z(:));
R=zeros(size(Z));
% R1=zeros(size(Z));
nn = 1;
for method=1:1:nn
    for i=1:3
        if method==1            %zm�㷨
            [R(:,:,i),L]=zm_retinex_Light1(Z(:,:,i),a,b,c,icount,0);
        elseif method==2        %mccann�㷨
            R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
        elseif method==3        %Kimmel�㷨����1
%             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5); %���һ��������Ч
            R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,1,4.5); %���һ��������Ч
        elseif method==4        %Kimmel�㷨����2
            R(:,:,i)=Z(:,:,i)-zm_Devir_retinex(Z(:,:,i),a,b);
        end
    end
    
    m=mean2(R);s=std(R(:));
    mini=max(m-step*s,min(R(:)));maxi=min(m+step*s,max(R(:)));
    range=maxi-mini;
    result=(R-mini)./range*0.8;
    %result=max(result(:))-result;  
    
%     sprintf('���ڴ���%s',imgFiles(1).name)
%     imwrite(result,[path,num2str(method),imgFiles(1).name]);
%     imwrite(result,[path,'֡ƽ��512retinex_Light','.jpg']);
%     figure,imshow(result); title(['ȥ���RGB��ǿͼ',num2str(method)]);
end
time1=toc
figure,imshowpair(img,result,'montage')
result1=uint8(result*255);
imwrite(result1,[path,'֡ƽ��zm_retinex480-640-1.jpg']);
gray = rgb2gray(result1);
figure(2),imhist(gray)
axis tight;
set(gca,'LooseInset',get(gca,'TightInset'))
set(gca,'looseInset',[0 0 0 0])

%%
quality=testquality(result1,1);

% ssimval = ssim(result1,imgpwer);
% Psnr=psnr(result1,imgpwer);  
% str=sprintf('ssim:%.4f psnr:%.4f',ssimval,Psnr);  %�������Ϊchar
% display(str)

value=[quality time1];
% value=[quality ssimval Psnr time1];
value=roundn(value,-4);  %������λС�� b=vpa(value,4)������Ч����

mRowRange='R5:W5';
writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\ѧϰ\С����\data.xlsx',value,1,mRowRange);

