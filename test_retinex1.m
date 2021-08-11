%----------------------------retinex低照度图像增强-帧平均-----------------------%
% clc
clear 
% path='G:\照片\6.16\';
% DIR='G:\照片\6.16\576-704\';
path='D:\Cammera\7.17\7.17result\';
DIR='D:\Cammera\7.17\7.17-480-640\';
% path='G:\照片\7.2\';
% DIR='G:\照片\7.2\1\576-704-1\';
imgFiles = dir([DIR,'*.jpg']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
tic

[N, ~]= size(imgFiles);
I=imread([DIR ,imgFiles(1).name]);
img=im2double(I);
% figure,imshow(img),title('原低照图')

% A=zeros(size(img));
for k=2:N
    A=im2double(imread([DIR ,imgFiles(k).name]));
    img=img+A;
%     sprintf('正在读取%s',imgFiles(k).name)
end
img=img./N;

% figure;imshow(I),title('原低照度图帧平均');
% imwrite(img,[path,'原低照度帧平均512','.jpg']);
%%
%------------------power伽马增强-------------%
% Iomean=90/255;
% % img_mean=mean2(img);
% img_mean=sum(median(median(img)))/3;
% p=log(Iomean)/log(img_mean)+0.005;
% imp=power(img,p); %pmean=0.3184;
% 
% time=toc
% figure,imshow(imp)
% quality=testquality(uint8(round(imp*255)),1);
% imwrite(imp,[path(1:15),'帧平均power480-640.jpg']);
% % imwrite(imp,[path,'帧平均576-704power','.jpg']);
% % % qualityscore = SSEQ(imp)
% 
% % quality_orig=testquality(I);

%%
% %------------------retinex SSR MSR MSRCR伽马增强-------------%
% method=2;
% switch method 
%     case 1
%         imp=SSR(img,80);
%         imwrite(imp,[path,'帧平均SSR','.jpg']);
%     case 2
%         imp=MSR(img,10,80,200);
%         imwrite(imp,[path,'帧平均MSR','.jpg']);
%     case 3
%         imp=MSRCR(img,10,80,200);
%         imwrite(imp,[path,'帧平均MSRCR','.jpg']);
% end
% % time=toc
% figure,imshow(imp)
% quality1=testquality(uint8(round(imp*255)));
% 
% % y=imread([path,'帧平均576-704retinex','.jpg']);
% % quality=testquality(y);
% % quality_orig=testquality(I);
% % qualityscore = SSEQ(y)


%%
% %----------------retinex增强------------------%
% %     f = 512 / max(size(I));
% %     I = imresize(I,f);
% %     I(I < 0) = 0;I(I > 1) = 1;
% 
% % a=0.01;b=0.2;c=1; 
% % icount=8;
% a=0.008;b=0.3;c=1;
% icount=4;
% step=2.4;%decide the times  %step=2.2;
% 
% I=img+1;
% Z=log2(I); %Z=log(I);
% Z=Z/max(Z(:));
% R=zeros(size(Z));
% % R1=zeros(size(Z));
% nn = 1;
% for method=1:1:nn
%     for i=1:3
%         if method==1            %zm算法
%             [R(:,:,i),L]=zm_retinex_Light1(Z(:,:,i),a,b,c,icount,0);
%         elseif method==2        %mccann算法
%             R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
%         elseif method==3        %Kimmel算法程序1
% %             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5); %最后一个参数无效
%             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,1,4.5); %最后一个参数无效
%         elseif method==4        %Kimmel算法程序2
%             R(:,:,i)=Z(:,:,i)-zm_Devir_retinex(Z(:,:,i),a,b);
%         end
%     end
%     
%     m=mean2(R);s=std(R(:));
%     mini=max(m-step*s,min(R(:)));maxi=min(m+step*s,max(R(:)));
%     range=maxi-mini;
%     result=(R-mini)/range*0.8;
%     %result=max(result(:))-result;  
%     
% %     sprintf('正在处理%s',imgFiles(1).name)
% %     imwrite(result,[path,num2str(method),imgFiles(1).name]);
% %     imwrite(result,[path,'帧平均512retinex_Light','.jpg']);
%     figure,imshow(result); title(['去噪后RGB增强图',num2str(method)]);
% end
% time1=toc
% % figure,
% % subplot(121),imshow(img),title('原低照度帧平均');
% % subplot(122),imshow(result),title('帧平均512retinex_Light');
% quality=testquality(uint8(round(result*255)));
% 
% % imwrite(result,[path,'帧平均576-704retinex_Light','.jpg']);
% 
% % qualityscore = SSEQ(y)
% % % qualityscore = SSEQ(result)

%%
% tic
% for j = 2:N
%     B=imread([DIR ,imgFiles(j).name]);
%     I1=im2double(B);
%     I1=I1+1;
%     Z1=log(I1);
%     Z1=Z1/max(Z1(:));
%     
%     R1=Z1-L;
%     m1=mean2(R1);s1=std(R1(:));
%     mini1=max(m1-step*s1,min(R1(:)));maxi1=min(m1+step*s1,max(R1(:)));
%     range1=maxi1-mini1;
%     result1=(R1-mini1)/range1*0.8;
%     %result1=max(result1(:))-result1;
%     
%     sprintf('正在处理%s',imgFiles(j).name)
%     imwrite(result,[path,'retinex_Light',num2str(j),'.jpg']);
% %     imwrite(result1,[path,num2str(method),imgFiles(j).name]);
% end
% time2=toc


