import numpy as np
import scipy.signal
import scipy.io
from pathlib import Path
import mne
import logging # 使用日志记录更清晰

# --- 配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义基础路径 (请根据你的实际路径修改)
BASE_DATA_DIR = Path('D:/Power/2.2extract-asr/TOM/')
BASE_RESULT_DIR = Path('D:/Power/result/')
# 确保结果目录存在
BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 定义组信息
GROUPS = {
    "TD": {
        "data_dir": BASE_DATA_DIR / "TD",
        "output_file": BASE_RESULT_DIR / "TD_power_tom.mat"
    },
    "ASD": {
        "data_dir": BASE_DATA_DIR / "ASD",
        "output_file": BASE_RESULT_DIR / "ASD_power_tom.mat"
    }
}

# 定义频带 (使用字典更清晰)
BANDS = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    # "Gamma": (35, 45) # MATLAB代码中定义但未在 1:3 循环中使用
}
# 定义处理顺序和用于索引结果数组的列表
BANDS_ORDERED = ["Theta", "Alpha", "Beta"]
N_BANDS = len(BANDS_ORDERED) # 频带数量

# 定义信号处理参数
FS_EXPECTED = 500  # 预期的采样率 (最好从文件中读取确认)
WINDOW_DURATION_SEC = 1.0 # 窗口时长，单位秒
WINDOW_OVERLAP_RATIO = 0.5 # 重叠比例

# --- 新增配置开关 ---
EXPAND_EPOCHS = False # True: 将每个epoch视为独立样本; False: 平均所有epoch为一个样本 (原始行为)
# --- !!! 新增参数：限制每个被试处理的Epoch数量 !!! ---
# 仅在 EXPAND_EPOCHS = True 时生效。如果设为 None，则处理所有 epochs。
MAX_EPOCHS_PER_SUBJECT = 30

logging.info(f"Epoch 处理模式: {'每个Epoch视为独立样本' if EXPAND_EPOCHS else '平均所有Epoch'}")
if EXPAND_EPOCHS:
    if MAX_EPOCHS_PER_SUBJECT is not None and MAX_EPOCHS_PER_SUBJECT > 0:
        logging.info(f"每个被试最多处理前 {MAX_EPOCHS_PER_SUBJECT} 个 Epochs。")
    elif MAX_EPOCHS_PER_SUBJECT is None:
         logging.info(f"将处理每个被试的所有 Epochs (MAX_EPOCHS_PER_SUBJECT = None)。")
    else:
        logging.warning(f"MAX_EPOCHS_PER_SUBJECT ({MAX_EPOCHS_PER_SUBJECT}) 设置无效，将处理每个被试的所有 Epochs。")
        MAX_EPOCHS_PER_SUBJECT = None # 修正为处理所有

# --- 主处理函数 ---
def calculate_band_power(data_dir: Path, output_file: Path, group_name: str):
    """
    计算指定目录下所有.set文件的各频带功率。

    Args:
        data_dir (Path): 包含.set文件的目录路径。
        output_file (Path): 保存结果的.mat文件路径。
        group_name (str): 组名（用于日志记录）。
    """
    logging.info(f"--- 开始处理组: {group_name} ---")
    logging.info(f"数据目录: {data_dir}")
    logging.info(f"输出文件: {output_file}")

    # 查找所有 .set 文件
    set_files = sorted(list(data_dir.glob('*.set')))
    if not set_files:
        logging.warning(f"在 {data_dir} 中未找到 .set 文件。跳过该组。")
        return

    n_subjects = len(set_files)
    logging.info(f"找到 {n_subjects} 个 .set 文件。")

    # 初始化结果存储
    subject_filenames = [] # 记录处理的文件名
    all_results = [] # 用于存储每个样本（被试或epoch）的结果
    subject_map = [] # 如果 EXPAND_EPOCHS=True, 记录每个结果对应的原始被试索引
    epoch_map = []   # 如果 EXPAND_EPOCHS=True, 记录每个结果对应的原始epoch索引 (在被试内)
    channel_names = None # 存储通道名称

    # 遍历每个被试 (.set 文件)
    for m, file_path in enumerate(set_files):
        subject_filenames.append(file_path.name)
        logging.info(f"正在处理文件 {m + 1}/{n_subjects}: {file_path.name}")

        try:
            # 加载 EEGLAB .set 文件
            epochs = mne.io.read_epochs_eeglab(file_path, verbose=False)
            epochs.load_data() # 将数据加载到内存

            # 第一次加载文件时获取通道名
            if channel_names is None:
                channel_names = epochs.ch_names
                logging.info(f"检测到 {len(channel_names)} 个通道: {', '.join(channel_names)}")
            elif channel_names != epochs.ch_names:
                logging.warning(f"文件 {file_path.name} 的通道名称与之前的不同！将使用第一个文件的通道名称。")
                # 这里可以根据需要决定是报错停止还是继续

            # 获取采样率并验证
            fs = epochs.info['sfreq']
            if abs(fs - FS_EXPECTED) > 1e-3:
                logging.warning(f"文件 {file_path.name} 的采样率 ({fs} Hz) 与预期 ({FS_EXPECTED} Hz) 不同。将使用文件自身的采样率 {fs} Hz 进行计算。")
            # fs = FS_EXPECTED # 强制使用预期采样率？或者使用文件自带的？当前代码使用文件自带的
            # 如果需要强制，取消上面一行的注释，并注释掉 else 分支

            # 获取数据: MNE 默认是 (n_epochs, n_channels, n_times) in Volts
            eeg_data_epochs = epochs.get_data(picks='eeg') # 确保只选择EEG通道
            n_epochs_total, n_channels, n_times_epoch = eeg_data_epochs.shape

            # --- !!! 关键修改：将单位从 V 转换为 μV !!! ---
            # 检查数据幅度，如果看起来像伏特（例如，绝对值远小于1），则转换
            # 这是一个启发式方法，更可靠的是了解数据的来源
            # if np.max(np.abs(eeg_data_epochs)) < 1:
            logging.debug("Data magnitude suggests Volts, converting to microvolts.")
            eeg_data_epochs = eeg_data_epochs * 1e6 # 现在是 (n_epochs, n_channels, n_times) in μV
            # else:
            #     logging.debug("Data magnitude suggests already in microvolts or other large unit, no conversion applied.")

            logging.debug(f"数据维度: {n_epochs_total} epochs, {n_channels} channels, {n_times_epoch} points/epoch.")

            # --- 计算 Welch 功率谱 ---
            # 定义 Welch 参数 (根据当前文件的 fs)
            window_length = int(fs * WINDOW_DURATION_SEC)
            overlap_length = int(window_length * WINDOW_OVERLAP_RATIO)
            # nfft = window_length # 或者使用 nextpow2
            nfft = int(2**np.ceil(np.log2(window_length))) # 匹配 MATLAB 的 nextpow2
            window = scipy.signal.windows.hann(window_length)
            epsilon = np.finfo(float).eps # 防止 log10(0)

            if not EXPAND_EPOCHS:
                # --- 原始逻辑：平均所有 Epoch ---
                if n_epochs_total == 0:
                    logging.warning(f"文件 {file_path.name} 包含 0 个 epochs，无法计算平均功率，跳过此文件。")
                    # 从 subject_filenames 中移除，因为它没有贡献数据
                    subject_filenames.pop()
                    continue # 跳到下一个文件

                # 转置为 (n_channels, n_epochs, n_times) 再 reshape 为 (n_channels, n_epochs * n_times)
                eeg_data_concatenated = eeg_data_epochs.transpose(1, 0, 2).reshape(n_channels, -1)
                logging.debug(f"合并 Epochs 后数据维度: {eeg_data_concatenated.shape}")

                subject_power = np.zeros((n_channels, N_BANDS)) # 存储当前被试的结果

                # 遍历每个通道
                for chan_idx in range(n_channels):
                    data_chan = eeg_data_concatenated[chan_idx, :] # μV 单位

                    # 计算 Welch PSD (对连接后的数据)
                    frequencies, psd = scipy.signal.welch(
                        data_chan,
                        fs=fs,
                        window=window,
                        nperseg=window_length,
                        noverlap=overlap_length,
                        nfft=nfft,
                        scaling='density', # 结果是 μV^2/Hz
                        average='mean'
                    )

                    # 遍历每个频带
                    for band_idx, band_name in enumerate(BANDS_ORDERED):
                        low_edge, high_edge = BANDS[band_name]
                        freq_indices = np.where((frequencies >= low_edge) & (frequencies <= high_edge))[0]

                        if len(freq_indices) > 0:
                            psd_in_band = psd[freq_indices] # μV^2/Hz
                            # 计算平均功率 (dB)
                            mean_power_db = np.mean(10 * np.log10(psd_in_band + epsilon)) # dB re μV^2/Hz
                        else:
                            logging.warning(f"文件 {file_path.name}, 通道 {chan_idx+1}, 频带 {band_name} 未找到频率点。功率设为 NaN。")
                            mean_power_db = np.nan
                        subject_power[chan_idx, band_idx] = mean_power_db

                all_results.append(subject_power) # 添加当前被试的结果 (n_channels, n_bands)

            else:
                # --- 新逻辑：每个 Epoch 视为独立样本 (带截断) ---

                # 确定要处理的epoch数量
                if n_epochs_total == 0:
                    logging.warning(f"文件 {file_path.name} 包含 0 个 epochs，跳过此文件的 epoch 处理。")
                    continue # 跳到下一个文件处理循环

                if MAX_EPOCHS_PER_SUBJECT is not None and MAX_EPOCHS_PER_SUBJECT > 0:
                    epochs_to_process = min(n_epochs_total, MAX_EPOCHS_PER_SUBJECT)
                    if epochs_to_process < n_epochs_total:
                        logging.info(f"被试 {file_path.name}: 共有 {n_epochs_total} 个 epochs, 将只处理前 {epochs_to_process} 个。")
                    else:
                         logging.debug(f"被试 {file_path.name}: 处理所有 {n_epochs_total} 个 epochs (未达到上限 {MAX_EPOCHS_PER_SUBJECT})。")
                else: # 处理所有 epochs (MAX_EPOCHS_PER_SUBJECT is None or <= 0)
                    epochs_to_process = n_epochs_total
                    logging.debug(f"被试 {file_path.name}: 处理所有 {n_epochs_total} 个 epochs。")

                # 遍历选定的 Epochs
                for epoch_idx in range(epochs_to_process): # <--- 使用计算出的 epochs_to_process
                    epoch_power = np.zeros((n_channels, N_BANDS)) # 存储当前 epoch 的结果

                    # 遍历每个通道
                    for chan_idx in range(n_channels):
                        # 获取单个 epoch, 单个通道的数据 (μV)
                        data_epoch_chan = eeg_data_epochs[epoch_idx, chan_idx, :]

                        # 计算 Welch PSD (对单个 epoch 的数据)
                        frequencies, psd = scipy.signal.welch(
                            data_epoch_chan,
                            fs=fs,
                            window=window,
                            nperseg=window_length,
                            noverlap=overlap_length,
                            nfft=nfft,
                            scaling='density', # 结果是 μV^2/Hz
                            average='mean' # 对单个 epoch 内的窗口进行平均
                        )

                        # 遍历每个频带
                        for band_idx, band_name in enumerate(BANDS_ORDERED):
                            low_edge, high_edge = BANDS[band_name]
                            freq_indices = np.where((frequencies >= low_edge) & (frequencies <= high_edge))[0]

                            if len(freq_indices) > 0:
                                psd_in_band = psd[freq_indices] # μV^2/Hz
                                # 计算平均功率 (dB)
                                mean_power_db = np.mean(10 * np.log10(psd_in_band + epsilon)) # dB re μV^2/Hz
                            else:
                                logging.warning(f"文件 {file_path.name}, Epoch {epoch_idx+1}, 通道 {chan_idx+1}, 频带 {band_name} 未找到频率点。功率设为 NaN。")
                                mean_power_db = np.nan
                            epoch_power[chan_idx, band_idx] = mean_power_db

                    all_results.append(epoch_power) # 添加当前 epoch 的结果 (n_channels, n_bands)
                    subject_map.append(m)           # 记录对应的被试索引
                    epoch_map.append(epoch_idx)     # 记录对应的 epoch 索引 (在被试内, 0 到 epochs_to_process-1)

        except Exception as e:
            logging.error(f"处理文件 {file_path.name} 时发生错误: {e}", exc_info=True)
            # 如果出错，从 subject_filenames 中移除，因为它没有贡献有效数据
            # 注意：如果 EXPAND_EPOCHS=True 且部分 epochs 已处理，这可能导致 subject_map/epoch_map 与 all_results 不完全匹配
            # 一个更健壮的方法可能是记录错误，并在保存时过滤掉与错误文件相关的结果，但这会增加复杂性。
            # 当前实现：如果文件处理中途出错，则该文件的所有已处理/未处理 epoch 结果都不会包含在最终输出中。
            # 需要确保在出错时，已添加到 all_results/subject_map/epoch_map 的部分数据被移除，或者在最后保存前进行过滤。
            # 为了简化，我们假设如果 try 块中出错，则该文件的所有内容都无效。
            # 如果在 EXPAND_EPOCHS=True 模式下，错误发生在某个 epoch 处理中，需要移除已添加的该文件的所有 epoch 数据。
            # 这里我们先移除文件名，并在保存前检查 all_results 是否为空。
            subject_filenames.pop() # 移除出错的文件名
            # 清理可能已部分添加的数据 (如果 EXPAND_EPOCHS=True)
            if EXPAND_EPOCHS:
                # 找到与当前被试索引 m 相关的所有条目并移除
                indices_to_remove = [i for i, subj_idx in enumerate(subject_map) if subj_idx == m]
                if indices_to_remove:
                    logging.warning(f"由于文件 {file_path.name} 出错，将移除已处理的 {len(indices_to_remove)} 个 epochs 的结果。")
                    # 从后往前删除以避免索引混乱
                    for i in sorted(indices_to_remove, reverse=True):
                        del all_results[i]
                        del subject_map[i]
                        del epoch_map[i]

    # --- 准备并保存结果 ---
    if not all_results:
        logging.warning(f"{group_name} 组没有成功处理任何数据，未保存文件。")
        return

    logging.info(f"完成 {group_name} 组处理。正在准备保存结果...")

    # 将列表中的结果堆叠成一个 NumPy 数组
    # 最终数组维度：
    # - EXPAND_EPOCHS=False: (n_successful_subjects, n_channels, n_bands)
    # - EXPAND_EPOCHS=True:  (total_successful_epochs, n_channels, n_bands)
    power_final = np.stack(all_results, axis=0)

    logging.info(f"最终结果数组维度: {power_final.shape}")

    # 修正 subject_filenames 列表，只包含成功处理的文件
    # 如果 EXPAND_EPOCHS=False，subject_filenames 长度应与 power_final.shape[0] 匹配
    # 如果 EXPAND_EPOCHS=True，subject_map 中的最大索引 + 1 应等于 len(subject_filenames)
    # （上面的错误处理逻辑已尝试维护 subject_filenames 的一致性）

    save_dict = {
        'power_data': power_final,
        'band_names': BANDS_ORDERED,
        'channel_names': channel_names if channel_names else "Unknown", # 保存通道名
        'subject_files': subject_filenames, # 只包含成功处理的原始文件名列表
        'processing_mode': 'average_epochs' if not EXPAND_EPOCHS else 'expand_epochs'
    }

    if EXPAND_EPOCHS:
        # 如果扩展了 epochs，添加映射信息
        save_dict['subject_map'] = np.array(subject_map) # 每个样本对应的被试索引 (基于成功处理的 subject_filenames 列表)
        save_dict['epoch_map'] = np.array(epoch_map)     # 每个样本对应的被试内epoch索引 (0 to epochs_to_process-1)
        # 生成每个样本对应的文件名列表，确保映射正确
        try:
            save_dict['subject_file_map'] = [subject_filenames[i] for i in subject_map]
        except IndexError:
             logging.error("创建 subject_file_map 时发生索引错误！subject_map 和 subject_filenames 可能不匹配。")
             save_dict['subject_file_map'] = ["Error creating map"] # 标记错误

        logging.info(f"已添加 subject_map (shape: {save_dict['subject_map'].shape}) 和 epoch_map (shape: {save_dict['epoch_map'].shape}) 到输出文件。")
        # 验证映射关系
        if len(save_dict['subject_file_map']) != power_final.shape[0]:
             logging.warning("警告：subject_file_map 的长度与 power_data 的第一维不匹配！")
        if save_dict['subject_map'].size > 0 and save_dict['subject_map'].max() >= len(subject_filenames):
             logging.warning("警告：subject_map 中的索引超出了 subject_files 列表的范围！")

    else:
        # 如果是平均模式，第一维直接对应 subject_files
        logging.info(f"结果数组的第一维 ({power_final.shape[0]}) 对应 'subject_files' 列表 ({len(subject_filenames)} 个被试)。")
        if power_final.shape[0] != len(subject_filenames):
             logging.warning("警告：power_data 的第一维与 subject_files 列表的长度不匹配！")

    logging.info(f"正在保存结果到 {output_file}...")
    try:
        scipy.io.savemat(output_file, save_dict, long_field_names=True) # long_field_names=True 兼容长变量名
        logging.info(f"结果已成功保存。")
    except Exception as e:
        logging.error(f"保存结果到 {output_file} 时发生错误: {e}", exc_info=True)

    logging.info(f"--- 完成处理组: {group_name} ---")

# --- 执行处理 ---
if __name__ == "__main__":
    for group_name, config in GROUPS.items():
        calculate_band_power(
            data_dir=config["data_dir"],
            output_file=config["output_file"],
            group_name=group_name
        )
    logging.info("所有组处理完成。")