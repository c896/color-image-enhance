% clc;
%----------------YUV-full-------------%
clear ;
close all;
% DIR='D:\Cammera\7.17\7.17-480-640\'; %均值0.0174  4.437  不测
% path='D:\Cammera\7.17\7.17result\';

for zu=1:6
    
    [DIR,path,xlspath]=Zu(zu);
    sprintf('当前第%d组，path:%s',zu,DIR)
    img=imread([path,'帧平均power480-640.jpg']);
    
    %% 设置初始参数
    if ismember(zu,[0 4 5])
        imgFiles = dir([DIR,'*.jpg']);
    elseif ismember(zu,[1 2 3 6 7])
        imgFiles = dir([DIR,'*.bmp']);
    end
    % [N, ~]= size(imgFiles);
    N=200;
    colorspace = ' ';
    method='power';
    num0=25;
    %     num0=1:1:200;
    for ii=1:numel(num0)
        
        close all;
        num=num0(ii)  %分组数
        % num=ii;%分组数
        % num=5;%分组数  [floor(200/13) mod(200,13)]
        
        imgnum=ones(1,num)*floor(200/num);%帧叠加数
        
        r=0.1;%尺寸比例变换系数
        
        % N2=floor(N/imgnum(1));
        %floor向负无穷取整 fix向0取整  ceil向正无穷取整
        %rem(N,imgnum) 当x和y的正负号一样的时候，两个函数结果是等同的；当x和y的符号不同时，rem函数结果的符号和x的一样，而mod和y一样。
        N1=mod(N,imgnum(1)*num);
        n=1;
        while N1~=0
            imgnum(n)=imgnum(n)+1;
            n=n+1;
            N1=N1-1;
        end
        % sum(imgnum)
        %% 预处理
        % I1= imread([DIR , imgFiles(1).name]);
        % I2= imread([DIR , imgFiles(101).name]);
        I= imread([DIR , imgFiles(100).name]);
        % I=(I1+I2+I3)./3;
        I_orig = im2double(I);
        Imean=mean2(I_orig);
        
        if Imean<=0.01
            Imean=mean2(I_orig);
            p1=log(0.4);
            beta=0.02; %0.02
        elseif Imean<=0.02
            p1=log(round(Imean*1000)/100+0.3);
            beta=0.02;
        else
            p1=log(0.5);%0.53
            beta=0.02;%0.05
        end
        
        [width,height,m]=size(I_orig);
        switch method
            case 'power'
                %----------------------------1-----------------------------------%
                p=p1/log(Imean); %p=round(10*p1/log(Imean))/10;%  p=0.3;
                Img=power(I_orig,p);  %imshow(Img) mean2(Img)
                %         Img=imdivide(im2double(img),10);
                %         Img=power(Img,p);
%                 I1=rgb_yuv(im2double(Img),colorspace);
%                 I1 = reshape(I1, [width, height, 3]);
            case 'retinex'
                Img=retinex1(I_orig);
                path='G:\照片\7.2\yuvresulta\';
            case 'other'
                Img=I_orig;
        end
        
        N2=width*height;
        
        %% 帧平均
        tic;
        sample1 = zeros(num,N2);
        sample2 = zeros(num,N2);
        sample3 = zeros(num,N2);
        Ic=zeros(width,height,3);
        % Ic=zeros(N1,1);
        for j = 1:num
            I_rgb=zeros(width,height,3);
            for k=1:imgnum(j)
                num1=imgnum(j)*(j-1)+k;
                Ii=im2double(imread([DIR ,imgFiles(num1).name]));
                I_rgb=I_rgb+Ii;
            end
            
            I_rgb=I_rgb./imgnum(j);
            
            %----------------------------1-----------------------------------%
            %         p=p1/log(mean2(I_rgb));
            p=p1/log(sum(median(median(I_rgb)))/3);
            %         P(j)=p;
            I_rgb=power(I_rgb,p); %I_rgb=log2(1+I_rgb);           
            Ic=Ic+I_rgb;

            sample1(j,:) = reshape(I_rgb(:,:,1),1,[]);
            sample2(j,:) = reshape(I_rgb(:,:,2),1,[]);
            sample3(j,:) = reshape(I_rgb(:,:,3),1,[]);
        end
        
        % imshow(I_rgb)
        
        Ic=Ic./(num);
%         Ic=reshape(Ic,width,height,3);
        % figure,imshow(Ic);
        
        %psnr_noise = 10*log10(peak^2/(mean((img11(:)-nimg1(:)).^2)));
        
        %% -----------------wasobi盲分离----------------------%%
        for k=1:m
            try
                %             [~, ~, ~,signals1]= iwasobi(sample1(),1,0.99);
                [~, ~, ~,signals1]= iwasobi(eval(['sample',num2str(k)]),1,0.99);
                %     [Wwasobi, Winit, ISR,eval(['signals',num2str(i)])]= iwasobi(sample(),1,0.99);
                %     signals = fastica(sample(),'numOfIC',1);
            catch
                continue
            end
            % for i=2:m
            %     x=eval(['sample',num2str(i)]);
            %     Xmean=mean(x,2);
            %     x=x-Xmean*ones(1,N1);
            %     Signals=Wwasobi*x+(Wwasobi*Xmean)*ones(1,N1);
            %     eval(['signals',num2str(i),'=', 'Signals',';']);
            % end
%             clear Signals
           
            % Icyuv = reshape(I_yuv, [width, height, 3]);% figure,imshow(Icyuv)
            
            imgrgb=Ic(:,:,k); %噪声图全部帧平均的yuv  mean2(imgrgb)
            %     imgrgb=Icyuv(:,:,i);%噪声图帧平均的yuv
            %     imgrgb=I1(:,:,i); %原图 imshow(I1)
            
            %--------------信号分离后排序------------------%
            [maxorg,minorg,index,ss]=signalsort1(signals1,imgrgb,num,'max',r);%'sort' 最大SSIM法
            %         [maxorg,minorg,index]=signalsort3(signals1,[width,height],num,'max',r);%'sort'最大方差法
            %         [simg,nsimg]=signalsort(eval(['signals',num2str(i)]),imgrgb,N);
            
            
            %     S=eval(['signals',num2str(i)]);
            %for k=1:number2   %k是要储存的分离层数量
            %H = S(index(N2+1-k),:);
            
            H = signals1(index,:);
            Hmin = min(H);
            Hmax = max(H);
            %     maxorg = maxorg/2+0.5;
            %     minorg = minorg/2;
            H = (H - Hmin)./(Hmax-Hmin)*(maxorg-minorg)+minorg;
            %     H = (H - Hmin)./(Hmax-Hmin);
            maxHH = max(H(:));
            simg = reshape(H,width,height);
            if ss(index)==1
                simg = maxHH-simg;
            end
            eval(['simg',num2str(k),'=', 'simg',';'])
        end
        clear sample1
        clear sample2
        clear sample3
        clear signals1
        clear simg
        clear H
        
        if ~exist('simg3','var')
            continue
        end
        %%
        %---------------------------2-----------------------------------%
        for k=1:m
            % figure,imhist(simg)
            Simg=adapthisteq(eval(['simg',num2str(k)]),'Distribution','uniform','ClipLimit',0.01); %'uniform'平坦 'rayleigh'钟形 'exponential'曲线
            Simg=0.5*Simg+0.5*eval(['simg',num2str(k)]);  %mean2(round(simg1*255))
            eval(['Simg',num2str(k),'=', 'Simg',';'])
            % figure,
            % subplot(121),imshow(simg)
            % subplot(122),imshow(simg1)
        end
        Simgout=cat(3,Simg1,Simg2,Simg3);
        simgout=cat(3,simg1,simg2,simg3);
        % smean=mean2(simg1);
        % % p2=log(round(smean*1000)/1000+beta); %0.4 0.5之间
        % p2=log(smean+beta);
        clear simg1
        clear simg2
        clear simg3
        clear Simg1
        clear Simg2
        clear Simg3
        clear Simg
        
        % 高斯滤波
        HSIZE= min(width,height);%高斯卷积核尺寸
        % HSIZE=20;
        q=sqrt(2);
        SIGMA1=15;%论文里面的c
        SIGMA2=80;
        SIGMA3=250;
        F1 = fspecial('gaussian',HSIZE,SIGMA1/q);
        F2 = fspecial('gaussian',HSIZE,SIGMA2/q) ;
        F3 = fspecial('gaussian',HSIZE,SIGMA3/q) ;
        gaus1= imfilter(simgout, F1, 'replicate');
        gaus2= imfilter(simgout, F2, 'replicate');
        gaus3= imfilter(simgout, F3, 'replicate');
        Iq=(gaus1+gaus2+gaus3)/3;    %多尺度高斯卷积，加权，权重为1/3
        % Iq=(Iq*255);%  figure,imshow(Iq,[]);title('gaus光照分量');
        %% 皮尔模型
        %---------------------------3-----------------------------------%
        % Imx=max(max(simg)); % mean2(round(simg*255))
        
        Beta1=[0.1605 0.0675 -25.7754];%专利版本
        % Beta1=[0.082 0.5949 -21.8727];
        for k=1:m
        for i=1:width
            for j=1:height
                deta=1-Simgout(i,j,k);
                %         if deta<=0.05
                %             simg1(i,j)=simg1(i,j)-0.05;
                %             p1(i,j)=(Imean-gaus(i,j))/Imean; %若光照值高于预设值，则降低原图亮度
                if deta<=0.3
                    %             simg1(i,j)=simg1(i,j)-0.16;
                    %             simg1(i,j)=simg1(i,j)-0.32+0.32./(1+exp(-12*deta.^1));
                    %             simg1(i,j)=simg1(i,j)-Beta(1)+Beta(2)./(Beta(3)+exp(-Beta(4)*deta.^Beta(5)));
                    Simgout(i,j,k)=Simgout(i,j,k)-Beta1(1)./(1+Beta1(2).*exp(-Beta1(3).*deta));
                    %             Imean(i,j)=0.5-0.5*(Beta(1)-Beta(2)./(Beta(3)+exp(-Beta(4)*deta.^Beta(5))));
                    %             p1(i,j)=(Imean-gaus(i,j))/Imean;
                end
            end
        end
        end
        
        smean=mean2(Simgout);
        % p2=log(round(smean*1000)/1000+beta); %0.4 0.5之间
        p2=log(smean+beta);
        
        %%
        %----------------------------4-----------------------------------%
        Iomean=exp(p2); %*ones(width,height);   % mean2(L) mean2(v)  level;  0.5
        p3=(Iomean-Iq)./Iomean; %原  %若光照值高于预设值，则降低原图亮度  imshow(Imean)
        % gama=(ones(size(p1))*0.5).^p1;  imshow(p1)
        % p=p2/log(median(median(simg)))-beta;
        p=p2/log(mean2(Simgout));
        gama=power(p,p3);%根据公式gamma校正处理，论文公式有误 mean2(gama)
        imgout1=power(Simgout,gama);   %mean2(round(simgout*255))
        % imshow(gama) imshow(simgout) % imshow(p1)
        
        %%

        %%
        %-------合成分离去噪图-------%
        
        ymin = min(min(imgout1));
        ymax = max(max(imgout1));
        imgout1 = 0.95.*(imgout1 - ymin)./(ymax-ymin);%*(maxorg-minorg)+minorg; %归一化
        %---------------------------3-----------------------------------%
        % p4=p2./log(mean2(imgout1))-0.01;
        % imgout2=power(imgout1,p4);  % mean2(round(imgout2*255))
        
        time=toc
        
        % figure,imshow(imgout1,'border','tight','initialmagnification','fit');
        % subplot(221),imshow(I_orig);title('原低照图');% subplot(222),imshow(Img);title('原低照增强图');
        % subplot(223),imshow(imgout);title(['yuv合成分离去噪图',method]);% subplot(224),imshow(imgout1);title(['转rgb合成分离去噪图',method]);
        img_final=uint8(imgout1*255);
        
        figure,imshow(img_final);title('本算法');
        % subplot(221),imshow(I);title('原图');
        % subplot(222),imshow(Img);title('原低照增强图');
        % subplot(223),imshow(img_final);title('本算法');
        % figure(1),imshowpair(I_orig,imgout1,'montage')
        
        %%
        imwrite(imgout1,[path,'RGB合成分离去噪图',num2str(num),'.jpg']);
        
        quality1=testquality(img_final,0);
        ssimval = ssim(img_final,img);
        Psnr=psnr(img_final,img);
        
        str1=sprintf('m2:%.4f  %.4f',Iomean,(Iomean)*255);  %输出类型为char
        str2=sprintf('SSIM:%.4f PSNR:%.4fdb',ssimval,Psnr);  %输出类型为char
        display([str2,'  ',str1])
        
        % quality_orig=testquality(I,0);%原图
        % ssimval = ssim(I,img);
        % Psnr=psnr(uint8(Img*255),img);%原低照增强图
        
        % imwrite(Img,[path,'原低照增强图','.jpg'])
        
        % qualityscore = SSEQ(y_est)
        % quality=testquality(uint8(round(Img*255)));
        %%
        % value=time;
        value=[quality1 ssimval Psnr time];
        value=roundn(value,-4);  %保留几位小数 b=vpa(value,4)保留有效数字
        
        % [tmp1,tmp2,tmpRaw]=xlsread('D:\学习\小论文\data.xlsx');
        range=203:-1:4; %range=14:-1:4;
        % range=33:-1:4; %range=14:-1:4;
        x=num2cell(range);
        
        mapObj=containers.Map(num2cell(1:1:200),x);
        % mapObj=containers.Map(num2cell([2:1:9,10:5:30,40:10:200]),x);
        % mapObj=containers.Map({2,4,5,8,10,20,25,40,50,100,200},x);
        % mapObj(2)
        % keys(mapObj)
        % values(mapObj)
        
        % mRowRange=sprintf('L%d:L%d',mapObj(num),mapObj(num));%全局时间
        % mRowRange=sprintf('K%d:K%d',mapObj(num),mapObj(num));%缩放时间
        mRowRange=sprintf('B%d:J%d',mapObj(num),mapObj(num));
%         writematrix(value,xlspath,'Sheet',1,'Range',mRowRange) % xlswrite('D:\学习\小论文\data.xlsx',value,1,mRowRange);
        pause(2)
        
    end
    %     clc
end

