Official code for the manuscript "Prompt Customization for Continual Learning"

trained model could be found in "https://pan.baidu.com/s/1vZIpDEgYh23lla59WQzfOQ?pwd=uigy 
提取码：uigy"


************appendix*************


% \section{Revised Experiment Results of CODA-P}
% \begin{table}[!t]
% 	\centering
% 	\caption{Performance $A_a$ of CODA-P tested on the previous tasks after completing the training phase of each task on Split Cifar-100 dataset. `Tt' means ${t}$-th task.}
%     \begin{tabular}{p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}}
%     \toprule
%     \multicolumn{1}{l}{T1} & \multicolumn{1}{l}{T2} & \multicolumn{1}{l}{T3} & \multicolumn{1}{l}{T4} & \multicolumn{1}{l}{T5} & \multicolumn{1}{l}{T6} & \multicolumn{1}{l}{T7} & \multicolumn{1}{l}{T8} & \multicolumn{1}{l}{T9} & \multicolumn{1}{l}{T10} \\
%     \midrule
%     99.3  & 93.3  & 89.8  & 88.5  & 86.8  & 83.9  & 84.2  & 84.2  & 82.8  & 81.9  \\
%           & 96.6  & 94.4  & 85.2  & 82.5  & 80.8  & 79.7  & 78.2  & 78.0  & 78.0  \\
%           &       & 96.1  & 95.8  & 94.0  & 92.7  & 91.7  & 91.0  & 90.4  & 90.5  \\
%           &       &       & 95.4  & 93.7  & 91.4  & 91.1  & 90.1  & 90.0  & 87.1  \\
%           &       &       &       & 93.3  & 93.1  & 92.7  & 92.9  & 92.5  & 90.8  \\
%           &       &       &       &       & 84.6  & 83.3  & 82.1  & 81.9  & 80.1  \\
%           &       &       &       &       &       & 88.6  & 87.7  & 86.2  & 85.0  \\
%           &       &       &       &       &       &       & 90.8  & 90.4  & 89.1  \\
%           &       &       &       &       &       &       &       & 94.3  & 91.9  \\
%           &       &       &       &       &       &       &       &       & 89.8  \\
%     \bottomrule
%     \end{tabular}%

% 	\label{tab:coda-cifar}%
% \end{table}%

% \begin{table}[!t]
% 	\centering
% 	\caption{Performance $A_a$ of CODA-P tested on the previous tasks after completing the training phase of each task on Split ImageNet-R dataset. `Tt' means ${t}$-th task.}
%     \begin{tabular}{p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}p{0.46cm}<{\centering}}
%     \toprule
%     \multicolumn{1}{l}{T1} & \multicolumn{1}{l}{T2} & \multicolumn{1}{l}{T3} & \multicolumn{1}{l}{T4} & \multicolumn{1}{l}{T5} & \multicolumn{1}{l}{T6} & \multicolumn{1}{l}{T7} & \multicolumn{1}{l}{T8} & \multicolumn{1}{l}{T9} & \multicolumn{1}{l}{T10} \\
%     \midrule
%     94.5  & 90.6  & 87.3  & 86.4  & 82.3  & 79.9  & 77.7  & 75.7  & 75.1  & 74.2  \\
%           & 81.8  & 79.9  & 78.5  & 74.6  & 73.0  & 72.5  & 69.6  & 68.0  & 67.3  \\
%           &       & 86.6  & 85.5  & 86.1  & 84.4  & 82.5  & 81.9  & 80.7  & 79.7  \\
%           &       &       & 77.4  & 74.6  & 73.5  & 72.2  & 69.7  & 69.2  & 67.8  \\
%           &       &       &       & 82.9  & 82.3  & 78.7  & 77.7  & 77.7  & 76.5  \\
%           &       &       &       &       & 80.0  & 78.6  & 76.5  & 75.4  & 74.8  \\
%           &       &       &       &       &       & 81.1  & 80.4  & 79.2  & 77.7  \\
%           &       &       &       &       &       &       & 80.8  & 79.7  & 78.0  \\
%           &       &       &       &       &       &       &       & 73.9  & 71.9  \\
%           &       &       &       &       &       &       &       &       & 74.7  \\
%     \bottomrule
%     \end{tabular}%
% 	\label{tab:coda-R}%
% \end{table}%

% \label{sec:CodaPA}
% Tab .\ref{tab:coda-cifar} and Tab. \ref{tab:coda-R} give the performance of CODA-P tested on the previous tasks after completing the training of each task on Split CIFAR-100 dataset and Split ImageNet-R dataset, respectively. We calculate the $A_a$ and $F$ for CODA-P according to the two tables, the  $A_a$ results are 86.41 and 74.26 and the $F$ results are 7.17 and 7.91, respectively.
