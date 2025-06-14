clear;clc
data_path = 'D:\Power\1rawdata\TD';
save_path = 'D:\Power\2.1extract\BASE\TD';
cd(data_path)
files = dir('*.set');
fn = {files.name};


for i = 1:length(fn)

    EEG = pop_loadset('filename',fn{i},'filepath',data_path);
    EEG = eeg_checkset( EEG ); 
    %%删除所有mark
    EEG.event = [];
    %TOM
    %EEG = pop_select(EEG, 'time', [64 70;78 86;100 104;156 172;218 228;230 240;272 286]);

    %BASELINE
    EEG = pop_select(EEG, 'time', [0 6;24 34;46 54]);
    
    EEG = pop_saveset( EEG, 'filename',fn{i},'filepath',save_path);
end
clear;



