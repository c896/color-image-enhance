% clc
clear 
close all

zu=9;
[DIR,path,xlspath]=Zu(zu);
sprintf('��ǰ��%d�飬path:%s',zu,DIR)
% imgpwer=imread([path,'֡ƽ��power480-640.jpg']);

if ismember(zu,[0 4 5])
    imgFiles = dir([DIR,'*.jpg']);
elseif ismember(zu,[1 2 3 6 7])
    imgFiles = dir([DIR,'*.bmp']);
end

I=imread([DIR ,imgFiles(101).name]); %ԭͼ
close all
% %% MSRCR --------------------------------
% imp1=retinex(DIR,path,xlspath,zu);
% close all
% % %% adaptgamma----------------------------
% imp2=adaptgamma(DIR,path,xlspath,zu);
% close all
% % %% zmretinex-----------------------------
% imp3=zmretinex(DIR,path,xlspath,zu);
% close all
% %% LIME
% imp4=testLIME(DIR,path,xlspath,zu);
% close all
% %% Dong
% imp5=testDong(DIR,path,xlspath,zu);

%%
% figure,
% subplot(221),imshow(imp1) 
% subplot(222),imshow(imp2) 
% subplot(223),imshow(imp3) 
% subplot(224),imshow(imgpwer) 

%% ԭͼ 
for zu=8:9
    % zu=1;
    [DIR,path,xlspath]=Zu(zu);
    sprintf('��ǰ��%d�飬path:%s',zu,DIR)
%     img=imread([path,'֡ƽ��power480-640.jpg']);
    
    if zu<5
        imgFiles = dir([DIR,'*.jpg']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
    else
        imgFiles = dir([DIR,'*.bmp']);
    end
    
    I= imread([DIR , imgFiles(101).name]); % imshow(I)
    quality_orig=testquality(I,1);%ԭͼ
%     ssimval = ssim(I,img);
    % Psnr=psnr(uint8(Img*255),img);%ԭ������ǿͼ
    value=quality_orig;
    value=roundn(value,-4);  %������λС�� b=vpa(value,4)������Ч����
    disp(value)
    writematrix(value,xlspath,'Sheet',1,'Range','B3:F3')
    % imwrite(Img,[path,'ԭ������ǿͼ','.jpg'])
    % qualityscore = SSEQ(y_est)
    % quality=testquality(uint8(round(Img*255)));
    pause(0.2)
end

%% ֱ��ͼ
% %------------------��ǿͼ----------------------%
% imp1=imread([path 'std\yuva�ϳɷ���ȥ��ͼ25.jpg']);
% gray = rgb2gray(imp1);
% % gray = imp1(:,:,1); %R
% % gray = imp1(:,:,1); %G
% % gray = imp1(:,:,1); %B
% [count1,x1]=imhist(gray);
% 
% figure(1),bar(x1,count1)  % stem(x,count)
% % figure(),imhist(gray)
% axis tight;
% set (gcf,'Position',[200,200,640,480]); %[left bottom width height]
% set(gca,'FontSize',20);
% % set(gca, 'XLim', [200 255]);%��Чxlim([0,200]);
% h1=xlabel({'Grayscale'},'FontSize',25,'Rotation',0);%h1=xlabel({'�Ҷȼ�'},'FontSize',18,'Rotation',0);
% h2=ylabel({'Pixel count'},'FontSize',25,'Rotation',90);%h2=ylabel({'��','��','��'},'FontSize',18,'Rotation',0);
% set(h1, 'Units', 'Normalized', 'Position', [0.5, -0.1, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
% set(h2, 'Units', 'Normalized', 'Position', [-0.04, 0.5, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
% % set(gca,'LooseInset',get(gca,'TightInset'))
% % set(gca,'looseInset',[0 0 0 0])   
% RemovePlotWhiteArea(gca,[-0.01 0.08 0 -0.02]);
% ax = gca;
% ax.YAxis.Exponent = 3;
% 
% % set(gca,'yTickLabel',num2str(get(gca,'yTick')','%.2f'))
% % set(gca,'ytick',get(gca,'yTick')'*2)
% 
% % print(gcf, '-djpeg', '-r300', ['D:\ѧϰ\С����\','10.09��ǿֱ��ͼ.bmp']);
% % ------------------ԭͼ----------------------%
% gray = rgb2gray(I);
% % figure(2),imhist(gray)
% [count2,x2]=imhist(gray);
% figure(2),bar(x2,count2)  % stem(x,count)
% axis tight;
% set (gcf,'Position',[200,200,640,480]); %[left bottom width height]
% set(gca,'FontSize',20);
% % set(gca, 'XLim', [200 255]);%��Чxlim([0,200]);
% h1=xlabel({'Grayscale'},'FontSize',25,'Rotation',0);%h1=xlabel({'�Ҷȼ�'},'FontSize',18,'Rotation',0);
% h2=ylabel({'Pixel count'},'FontSize',25,'Rotation',90);%h2=ylabel({'��','��','��'},'FontSize',18,'Rotation',0);
% set(h1, 'Units', 'Normalized', 'Position', [0.5, -0.1, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
% set(h2, 'Units', 'Normalized', 'Position', [-0.06, 0.5, 0]); % ��ֵ��ylabel ���ƣ���ֵ�� ���ƣ�0���м�
% set(gca,'LooseInset',get(gca,'TightInset'))
% set(gca,'looseInset',[0 0 0 0])
% ax = gca;
% ax.YAxis.Exponent = 4;
% % set(gca,'yTickLabel',num2str(get(gca,'yTick')'/100))
% % hAxes = gca;
% % hAxes.YRuler.SecondaryLabel.String = '��10^5';
% 
% % histogram(gray,'BinWidth',5)
% % print(gcf, '-djpeg', '-r300', ['D:\ѧϰ\С����\','10.09ԭͼֱ��ͼ.bmp']);
