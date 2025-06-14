clear;clc
%定义数据所在的路径
data_path = 'D:\Power\2.1extract\BASE\ASD';
%定义数据的保存路径
save_path = 'D:\Power\2.2extract-asr\BASE\ASD';
%将数据所在的路径定义为工作路径
cd(data_path)
%筛选当前路径下所有的set结尾的文件
files = dir('*.set');
%提取文件名
fn = {files.name};


%对于每个被试
for i = 1:length(fn)
    %导入set格式的数据
    EEG = pop_loadset('filename',fn{i},'filepath',data_path);
    EEG = eeg_checkset( EEG ); 
    %% ASR
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','on','Distance','Euclidian');
    try
        EEG = eeg_regepochs(EEG, 'recurrence', 1, 'limits',[0 1], 'rmbase',NaN);
    %保存set格式的数据
        EEG = pop_saveset( EEG, 'filename',fn{i},'filepath',save_path);
    end
end

clear;

