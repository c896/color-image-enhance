clc
clear 

%% 
% if exist('filename','var') ~= 1
%     [filename, user_canceled] = imgetfile;
%     if user_canceled, error('canceled'); end;
% end
% I = imread(filename);
% figure('name','Original'), imshow(I);
%%
% DIR='D:\Cammera\7.2\1\原\';%输入图片所在文件夹的路径
% path='G:\照片\7.2\1\576-704\';
% path='D:\Cammera\7.2\1\480-640-1\';

% DIR='D:\Cammera\MV-CE013-50UC7.17\';
% path='D:\Cammera\7.17-480-640\';

% DIR='D:\Cammera\9.28\9.28-1\';%输入图片所在文件夹的路径
% path='D:\Cammera\9.28\9.28-1-480-640\';

% DIR='D:\Cammera\10.09\2\';%输入图片所在文件夹的路径
% path='D:\Cammera\10.09\2-480-640\';

DIR='E:\pictures\1湖面2\'; %均值1 0.0524 13.3742
path='E:\pictures\1湖面2-480-640\';

% DIR='E:\pictures\3校门口\'; %均值1 0.0524 13.3742
% path='E:\pictures\3校门口-480-640\';


imgFiles = dir([DIR,'*.bmp']);%输入图像的格式  dir('')列出指定目录下所有子文件夹和文件
[N, ~]= size(imgFiles);
tic

%%
% %----------------视频读取---------------------%
% file_name='C:\Users\CG\Desktop\5.28\SL_MO_VID.mp4';
% v = VideoReader(file_name);
% vidWidth = v.Width;
% vidHeight = v.Height;
% 
% k = 1;
% while hasFrame(v)
% 
% frame=readFrame(v);
% imwrite(frame,strcat('D:\视频降噪\照片2\帧1\',num2str(k),'.jpg'));% 保存帧
% k = k+1;
% end
% frame=[];
% num_frame=k-1;

% v.CurrentTime = 0.6;
% k = 1; 
% while v.CurrentTime <= 1
%     frame=readFrame(v);
%     k = k+1;
% end
% frame=[];
% numFrame=k-1;

% numFrames = obj.NumberOfFrames;% 帧的总数
%-----------从特定时间开始读取视频帧-----------%
% v.CurrentTime = 0.5; %指定在距视频开头 2.5 秒的位置开始读取。
% currAxes = axes; %创建一个坐标区对象以显示帧。然后，继续读取和显示视频帧，直到没有可供读取的帧为止。
% while hasFrame(v)
%     vidFrame = readFrame(v);
%     image(vidFrame, 'Parent', currAxes);
%     currAxes.Visible = 'off';
%     pause(1/v.FrameRate);
% end

%%
% %-------------------读取并播放影片文件----------------------%
% mov = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
%     'colormap',[]); %创建影片结构体数组 mov。
% k = 1;
% while hasFrame(v)
%     mov(k).cdata = readFrame(v); %一次读取一帧，直到视频结束。
%     imshow(mov(k).cdata)
%     k = k+1;
% end
% %基于视频的宽度和高度确定图窗大小。然后，按照视频的帧速率播放一次影片。
% hf = figure;  
% set(hf,'position',[150 150 vidWidth vidHeight]);
% movie(hf,mov,1,60);

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

%%
%------------图片裁剪-----------------%

% rect=[105 361 2815 2303]; %[x起点 y起点 宽 高] 704*576
% rect=[980 1380 1279 959]; %[x起点 y起点 宽 高] 640*480  0.5
rect=[10 450 2559 1919]; %[x起点 y起点 宽 高] 640*480   0.25
% rect=[980 1380 1279 959]; %[x起点 y起点 宽 高]
for j = 1:N
        [A,map]=imread([DIR ,imgFiles(j).name]);

%         B=imcrop(A,rect);
%         C= imresize(B,0.25);
        C= imresize(A,0.5);
%         subplot(121),imshow(A);
%         rectangle('Position',rect,'LineWidth',2,'EdgeColor','r')%显示图像剪切区域
%         subplot(122),imshow(C);
        sprintf('正在处理%s',imgFiles(j).name)
        if j<10
            imwrite(C,[path,sprintf('%s-00%d.bmp',imgFiles(j).name(1:14),j)]);
        elseif j<100
            imwrite(C,[path,sprintf('%s-0%d.bmp',imgFiles(j).name(1:14),j)]);
        else
            imwrite(C,[path,sprintf('%s-%d.bmp',imgFiles(j).name(1:14),j)]);             
        end
%         imwrite(C,[path,imgFiles(j).name]);
%         imwrite(B,['C:\Users\CG\Desktop\三维重建\','边缘1','.png']);
end

% for j = 1:N
%     A=imread([DIR ,imgFiles(j).name]);
%     I=im2double(A);
%     f = 512 / max(size(I));  
%     I = imresize(I,f);
%     I(I < 0) = 0;I(I > 1) = 1;
% %     figure,imshow(I)
%     imwrite(I,[path,imgFiles(j).name]);
% end

%%
% %----------------------------retinex低照度图像增强------------------------%
% for j = 1:N
%     A=imread([DIR ,imgFiles(j).name]);
%     I=im2double(A);
% %     f = 512 / max(size(I));  
% %     I = imresize(I,f);
% %     I(I < 0) = 0;I(I > 1) = 1;
% 
% %     A= imresize(A,0.2);   
%        
%     
%     a=0.01;b=0.2;c=1; %a=0.008;b=0.3;c=1;
%     icount=8;   %icount=4; 
%     step=2.4;%decide the times  %step=2.2;
%     I=I+1;
%     Z=log(I);
%     Z=Z/max(Z(:));
%     R=zeros(size(Z));
%     % nn = 3;
%     nn = 1;
%     for method=1:1:nn
%         for i=1:3
%             if method==1            %zm算法
%                 R(:,:,i)=zm_retinex_Light(Z(:,:,i),a,b,c,icount,0);
%             elseif method==2        %mccann算法
%                 R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
%             elseif method==3        %Kimmel算法程序1
%                 R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5);
%             elseif method==4        %Kimmel算法程序2
%                 R(:,:,i)=Z(:,:,i)-zm_Devir_retinex(Z(:,:,i),a,b);
%             end
%         end
%         
%         m=mean2(R);s=std(R(:));
%         mini=max(m-step*s,min(R(:)));maxi=min(m+step*s,max(R(:)));
%         range=maxi-mini;
%         result1=(R-mini)/range*0.8;  
%         %result1=max(result1(:))-result1;
%         
%         sprintf('正在处理%s',imgFiles(j).name)
%         imwrite(result1,[path,num2str(method),imgFiles(j).name]);
% %         figure,imshow(result1); title(['去噪后RGB增强图',num2str(method)]);
%     end
% end

toc