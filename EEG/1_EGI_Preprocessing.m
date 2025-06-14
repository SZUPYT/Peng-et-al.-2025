%% Modified By SZU Doctor Peng

%% 数据格式格式转化
clear;clc

%定义数据所在的路径
data_path = 'D:\ASD\1rawdata';
%定义数据的保存路径
save_path = 'D:\ASD\2raw_set';
%将数据所在的路径定义为工作路径
cd(data_path)
%筛选当前路径下所有的mff结尾的文件
files = dir('*.mff');
%提取文件名
fn = {files.name};
%对于每个被试
for i = 1:length(fn)
   %导入原始数据
    fprintf(fn{i})
    % 构建完整的文件路径
    filePath = fullfile(data_path, fn{i});
    
    % 使用 pop_mffimport 函数导入数据
    EEG = mff_import(filePath);
    EEG = eeg_checkset( EEG ); 
    setname = [fn{i}(1:end-4),'set']; 
    %保存set格式的数据
    EEG = pop_saveset( EEG, 'filename',setname,'filepath',save_path);
end


%% 通道定位 去除无用电极 滤波 降采样 分段
clear;clc
%定义数据所在的路径
data_path = 'D:\ASD\2raw_set';
%定义数据的保存路径
save_path = 'D:\ASD\3preprocess_set';
%将数据所在的路径定义为工作路径
cd(data_path)
%筛选当前路径下所有的vhdr结尾的文件
files = dir('*.set');
%提取文件名
fn = {files.name};
%对于每个被试
for i = 1:length(fn)
    %导入set格式的数据
    EEG = pop_loadset('filename',fn{i},'filepath',data_path);
    EEG = eeg_checkset( EEG ); 
    %EEG.chanlocs(63).labels = 'M1';

    %通道定位，加载脑电帽通道配置文件
    % 设置 SFP 格式的通道位置文件路径
    sfpFile = 'D:\ASD\EGI_64_2-9years.sfp';
    
    % 使用 pop_chanedit 函数加载 SFP 文件
    EEG = pop_chanedit(EEG, 'load', {sfpFile, 'filetype', 'sfp'});

    %Cz参考
    EEG = pop_reref( EEG, 65);

    %% 去除无用电极，EGI眼电电极为61、62、63、64，双侧乳突电极E29 E47可能包含非脑电成分，如果要用它重参考就不要删
    EEG = pop_select( EEG,'nochannel',{'E29' 'E47' 'E61' 'E62' 'E63' 'E64'});

    %%滤波，使用FIR方法,为EEGLAB工具箱默认方法
    %高通滤波
    EEG = pop_eegfiltnew(EEG, 'locutoff',1,'plotfreqz',0);
    %低通滤波
    EEG = pop_eegfiltnew(EEG, 'hicutoff',100,'plotfreqz',0);
    %凹陷滤波
    EEG = pop_eegfiltnew(EEG, 'locutoff',48,'hicutoff',52,'revfilt',1,'plotfreqz',0);

    channel_num = EEG.nbchan;

    %% 滤波，使用IIR方法
    %高通滤波，滤波前移除直流偏移
    %EEG  = pop_basicfilter( EEG,  1:channel_num, 'Boundary', 'boundary', 'Cutoff',  0.1, 'Design', 'butter', 'Filter', 'highpass', 'Order',  2, 'RemoveDC', 'on' );
    %低通滤波
    %EEG  = pop_basicfilter( EEG,  1:channel_num, 'Boundary', 'boundary', 'Cutoff',  40, 'Design', 'butter', 'Filter', 'lowpass', 'Order',  2);
    %凹陷滤波
    %EEG  = pop_basicfilter( EEG,  1:channel_num, 'Boundary', 'boundary', 'Cutoff',  50, 'Design', 'notch', 'Filter', 'PMnotch', 'Order', 180 );
 
    %% 降采样,根据文献修改，这个EGI本身就是500Hz.....
    %EEG = pop_resample( EEG, 500);

    % %% 删除实验开始前的数据段
    % %先对数据进行初步分段，从Mark开始到记录结束为一段
    % % 设置要保留的MARK
    % eventLabel = 'bgin';
    % %将整个实验过程的数据从原始文件中提取出来,本实验中刺激持续时间为300秒
    EEG = pop_epoch(EEG, 'bgin', [0, 300], 'epochinfo', 'yes');

    % 使用pop_selectevent函数选择特定事件之后的数据段
    %EEG = pop_selectevent(EEG, 'type', eventLabel, 'deleteevents', 'off', 'deleteepochs', 'on');

    % 删除坏导先保存数据到文件夹3
    EEG = pop_saveset( EEG, 'filename',fn{i},'filepath',save_path);

    full_channel = {EEG.chanlocs.labels};

    %% 搜索坏导并删除，按需启用
    EEG_remove_bad_channge = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    
    channel_after_remove = {EEG_remove_bad_channge.chanlocs.labels};

    removed_channel = setdiff(full_channel, channel_after_remove);

    % 打开一个文本文件进行写入
    fileID = fopen('D:\ASD\4remove_bad_channel_manual\output.txt', 'a');
    % 写入 当前数据文件名 和一个空行
    fprintf(fileID, '%s\n\n', fn{i});
    % 将结果写入文本文件
    for words = 1:length(removed_channel)
        fprintf(fileID, '%s\n', removed_channel{words});
    end
    fprintf(fileID, '\n\n');
    % 关闭文本文件
    fclose(fileID);

    %% 对比第1个Dataset得出删除的电极点，将这些电极点进行插值
    EEG = pop_interp(EEG_remove_bad_channge, EEG.chanlocs, 'spherical');

    %保存set格式的数据
    EEG = pop_saveset( EEG, 'filename',fn{i},'filepath','D:\ASD\4remove_bad_channel_manual\');

    % 定义保留的主成分数量
    m = channel_num; % 原始电极数量
    n = length(removed_channel); %刚刚插值的电极数量

    %% run ICA,ICA前根据文献决定是否分段
    EEG = pop_runica(EEG, 'extended',1,'pca',m-n,'interupt','on');
    %保存set格式的数据
    EEG = pop_saveset( EEG, 'filename',fn{i},'filepath','D:\ASD\5run_ICA\');
    
    %% 使用ICLABEL自动去除眼电等成分
    EEG = pop_iclabel(EEG, 'default');

    EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.9 1;0.9 1;0.9 1;0.9 1;NaN NaN]);

    EEG = pop_subcomp( EEG, [], 0);

    %% 重参考
    EEG = pop_reref( EEG, [],'refloc',struct('labels',{'E65'},'description',{'VREF'},'X',{-1.1427e-31},'Y',{-6.2206e-16},'Z',{10.159},'identifier',{1001},'type',{'EEG'},'ref',{'E65'},'sph_theta',{-90},'sph_phi',{90},'sph_radius',{10.159},'theta',{90},'radius',{0},'urchan',{[]},'datachan',{0}));
    %EEG = pop_reref( EEG, [] );
    EEG = pop_saveset( EEG, 'filename',fn{i},'filepath','D:\ASD\6remove_bad_components_manual\');

end

clear;
eeglab;

