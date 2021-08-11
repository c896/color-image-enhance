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
% DIR='D:\Cammera\7.2\1\ԭ\';%����ͼƬ�����ļ��е�·��
% path='G:\��Ƭ\7.2\1\576-704\';
% path='D:\Cammera\7.2\1\480-640-1\';

% DIR='D:\Cammera\MV-CE013-50UC7.17\';
% path='D:\Cammera\7.17-480-640\';

% DIR='D:\Cammera\9.28\9.28-1\';%����ͼƬ�����ļ��е�·��
% path='D:\Cammera\9.28\9.28-1-480-640\';

% DIR='D:\Cammera\10.09\2\';%����ͼƬ�����ļ��е�·��
% path='D:\Cammera\10.09\2-480-640\';

DIR='E:\pictures\1����2\'; %��ֵ1 0.0524 13.3742
path='E:\pictures\1����2-480-640\';

% DIR='E:\pictures\3У�ſ�\'; %��ֵ1 0.0524 13.3742
% path='E:\pictures\3У�ſ�-480-640\';


imgFiles = dir([DIR,'*.bmp']);%����ͼ��ĸ�ʽ  dir('')�г�ָ��Ŀ¼���������ļ��к��ļ�
[N, ~]= size(imgFiles);
tic

%%
% %----------------��Ƶ��ȡ---------------------%
% file_name='C:\Users\CG\Desktop\5.28\SL_MO_VID.mp4';
% v = VideoReader(file_name);
% vidWidth = v.Width;
% vidHeight = v.Height;
% 
% k = 1;
% while hasFrame(v)
% 
% frame=readFrame(v);
% imwrite(frame,strcat('D:\��Ƶ����\��Ƭ2\֡1\',num2str(k),'.jpg'));% ����֡
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

% numFrames = obj.NumberOfFrames;% ֡������
%-----------���ض�ʱ�俪ʼ��ȡ��Ƶ֡-----------%
% v.CurrentTime = 0.5; %ָ���ھ���Ƶ��ͷ 2.5 ���λ�ÿ�ʼ��ȡ��
% currAxes = axes; %����һ����������������ʾ֡��Ȼ�󣬼�����ȡ����ʾ��Ƶ֡��ֱ��û�пɹ���ȡ��֡Ϊֹ��
% while hasFrame(v)
%     vidFrame = readFrame(v);
%     image(vidFrame, 'Parent', currAxes);
%     currAxes.Visible = 'off';
%     pause(1/v.FrameRate);
% end

%%
% %-------------------��ȡ������ӰƬ�ļ�----------------------%
% mov = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
%     'colormap',[]); %����ӰƬ�ṹ������ mov��
% k = 1;
% while hasFrame(v)
%     mov(k).cdata = readFrame(v); %һ�ζ�ȡһ֡��ֱ����Ƶ������
%     imshow(mov(k).cdata)
%     k = k+1;
% end
% %������Ƶ�Ŀ�Ⱥ͸߶�ȷ��ͼ����С��Ȼ�󣬰�����Ƶ��֡���ʲ���һ��ӰƬ��
% hf = figure;  
% set(hf,'position',[150 150 vidWidth vidHeight]);
% movie(hf,mov,1,60);

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

%%
%------------ͼƬ�ü�-----------------%

% rect=[105 361 2815 2303]; %[x��� y��� �� ��] 704*576
% rect=[980 1380 1279 959]; %[x��� y��� �� ��] 640*480  0.5
rect=[10 450 2559 1919]; %[x��� y��� �� ��] 640*480   0.25
% rect=[980 1380 1279 959]; %[x��� y��� �� ��]
for j = 1:N
        [A,map]=imread([DIR ,imgFiles(j).name]);

%         B=imcrop(A,rect);
%         C= imresize(B,0.25);
        C= imresize(A,0.5);
%         subplot(121),imshow(A);
%         rectangle('Position',rect,'LineWidth',2,'EdgeColor','r')%��ʾͼ���������
%         subplot(122),imshow(C);
        sprintf('���ڴ���%s',imgFiles(j).name)
        if j<10
            imwrite(C,[path,sprintf('%s-00%d.bmp',imgFiles(j).name(1:14),j)]);
        elseif j<100
            imwrite(C,[path,sprintf('%s-0%d.bmp',imgFiles(j).name(1:14),j)]);
        else
            imwrite(C,[path,sprintf('%s-%d.bmp',imgFiles(j).name(1:14),j)]);             
        end
%         imwrite(C,[path,imgFiles(j).name]);
%         imwrite(B,['C:\Users\CG\Desktop\��ά�ؽ�\','��Ե1','.png']);
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
% %----------------------------retinex���ն�ͼ����ǿ------------------------%
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
%             if method==1            %zm�㷨
%                 R(:,:,i)=zm_retinex_Light(Z(:,:,i),a,b,c,icount,0);
%             elseif method==2        %mccann�㷨
%                 R(:,:,i)=zm_retinex_mccann992(Z(:,:,i),icount);
%             elseif method==3        %Kimmel�㷨����1
%                 R(:,:,i)=zm_Kimmel(Z(:,:,i),a,b,icount,4.5);
%             elseif method==4        %Kimmel�㷨����2
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
%         sprintf('���ڴ���%s',imgFiles(j).name)
%         imwrite(result1,[path,num2str(method),imgFiles(j).name]);
% %         figure,imshow(result1); title(['ȥ���RGB��ǿͼ',num2str(method)]);
%     end
% end

toc