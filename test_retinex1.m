%----------------------------retinex���ն�ͼ����ǿ-֡ƽ��-----------------------%
% clc
clear 
% path='G:\��Ƭ\6.16\';
% DIR='G:\��Ƭ\6.16\576-704\';
path='D:\Cammera\7.17\7.17result\';
DIR='D:\Cammera\7.17\7.17-480-640\';
% path='G:\��Ƭ\7.2\';
% DIR='G:\��Ƭ\7.2\1\576-704-1\';
imgFiles = dir([DIR,'*.jpg']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
tic

[N, ~]= size(imgFiles);
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

% figure;imshow(I),title('ԭ���ն�ͼ֡ƽ��');
% imwrite(img,[path,'ԭ���ն�֡ƽ��512','.jpg']);
%%
%------------------power٤����ǿ-------------%
% Iomean=90/255;
% % img_mean=mean2(img);
% img_mean=sum(median(median(img)))/3;
% p=log(Iomean)/log(img_mean)+0.005;
% imp=power(img,p); %pmean=0.3184;
% 
% time=toc
% figure,imshow(imp)
% quality=testquality(uint8(round(imp*255)),1);
% imwrite(imp,[path(1:15),'֡ƽ��power480-640.jpg']);
% % imwrite(imp,[path,'֡ƽ��576-704power','.jpg']);
% % % qualityscore = SSEQ(imp)
% 
% % quality_orig=testquality(I);

%%
% %------------------retinex SSR MSR MSRCR٤����ǿ-------------%
% method=2;
% switch method 
%     case 1
%         imp=SSR(img,80);
%         imwrite(imp,[path,'֡ƽ��SSR','.jpg']);
%     case 2
%         imp=MSR(img,10,80,200);
%         imwrite(imp,[path,'֡ƽ��MSR','.jpg']);
%     case 3
%         imp=MSRCR(img,10,80,200);
%         imwrite(imp,[path,'֡ƽ��MSRCR','.jpg']);
% end
% % time=toc
% figure,imshow(imp)
% quality1=testquality(uint8(round(imp*255)));
% 
% % y=imread([path,'֡ƽ��576-704retinex','.jpg']);
% % quality=testquality(y);
% % quality_orig=testquality(I);
% % qualityscore = SSEQ(y)


%%
% %----------------retinex��ǿ------------------%
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
%         if method==1            %zm�㷨
%             [R(:,:,i),L]=zm_retinex_Light1(Z(:,:,i),a,b,c,icount,0);
%         elseif method==2        %mccann�㷨
%             R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
%         elseif method==3        %Kimmel�㷨����1
% %             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5); %���һ��������Ч
%             R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,1,4.5); %���һ��������Ч
%         elseif method==4        %Kimmel�㷨����2
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
% %     sprintf('���ڴ���%s',imgFiles(1).name)
% %     imwrite(result,[path,num2str(method),imgFiles(1).name]);
% %     imwrite(result,[path,'֡ƽ��512retinex_Light','.jpg']);
%     figure,imshow(result); title(['ȥ���RGB��ǿͼ',num2str(method)]);
% end
% time1=toc
% % figure,
% % subplot(121),imshow(img),title('ԭ���ն�֡ƽ��');
% % subplot(122),imshow(result),title('֡ƽ��512retinex_Light');
% quality=testquality(uint8(round(result*255)));
% 
% % imwrite(result,[path,'֡ƽ��576-704retinex_Light','.jpg']);
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
%     sprintf('���ڴ���%s',imgFiles(j).name)
%     imwrite(result,[path,'retinex_Light',num2str(j),'.jpg']);
% %     imwrite(result1,[path,num2str(method),imgFiles(j).name]);
% end
% time2=toc


