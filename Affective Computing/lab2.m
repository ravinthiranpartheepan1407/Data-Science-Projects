%Clear
clf;clc;clear all;
%Data Import
address='F:\Applied Informatics\volunteer1.mat';
address2='F:\Applied Informatics\volunteer5.mat';
delimiter=' ';
A=importdata(address, delimiter);
B=importdata(address2, delimiter);
Attention=A.VarName4(:,1);
AttentionB=B.VarName4(:,1);
Meditation=A.VarName5(:,1);
MeditationB=B.VarName5(:,1);
EEG=A.VarName14(:,1);
EEGB=B.VarName14(:,1);
%Attention and Smooth Attention
figure (1)
smoothAttention=conv(Attention, ones(1,3)/3);
plot(Attention, 'r');
hold on
plot(smoothAttention,'g');
ylabel('Attention');
xlabel('Time (s)');
legend('Attention', 'Smooth Attention');
hold off
%Meditation Signal
figure (2)
plot(Meditation, 'g', 'Linewidth', 3.5);
ylabel('Meditation');
xlabel('Time (s)');
legend('Meditation');
%EEG Voltage signal
figure (3)
for i=1:length(EEG)
%1.8=input voltage, 2000=gain, 4096=value range
voltageEEG(i)=EEG(i)*(1.8/4096)/2000;
end
plot(voltageEEG, 'b', 'Linewidth', 2.5);
ylabel('EEG (voltage)');
xlabel('Time (s)');
legend('EEG (v)');
%Smoothing - moving average 
smoothEEG=conv(EEG, ones(1,10)/10);
%Fourier transformation
n=length(EEG);
fEEG=fft(smoothEEG, n);
magnitude=abs(fEEG);
phase=angle(fEEG);
%Magnitude Spectrum
figure(4)
plot(magnitude);
ylabel('Magnitude');
xlabel('Hz');
legend('Signal');
%Phase Spectrum
figure(5)
plot(phase);
ylabel('Phase');
xlabel('Hz');
legend('Signal');
%Cross Correlation, correlation of average signal 
figure(6)
[r, lag]=xcorr(Attention(1:240)-mean(Attention), AttentionB(1:240)-mean(AttentionB), 240,'coeff');
plot(lag, r, 'c');
ylabel('Cross-Correlation');
xlabel('Samples');
legend('Cross-Correlation');
%Correlation-Pearson
R=corrcoef(Attention(1:200), AttentionB(1:200))
A=std(Attention);
A
sum=0;
N=length(Attention);
for i=1:N
    sum=sum+(Attention(i)-mean(Attention))^2;
end
A_notMatlab=sqrt(sum/(N-1))
%Cross-correlation Maximum
maxCorr=max(abs(r))
%Average of Attention
sum=0;
for i=1:length(Attention)-1;
    if(Attention(i)>30)
    sum=sum+Attention(i);
    end
end

AttentionAverage=sum/length(Attention)-1;
AttentionAverage
maximum=max(Attention);
%Sublot
figure(7)
subplot(2,2,[1,2])
plot(Attention, 'r');
ylabel('Attention');
xlabel('Time (s)');
legend('Attention');
title('Attention')
set(gca, 'fontweight', 'bold', 'fontsize', 16)
subplot(2,2,3)
plot(Meditation, 'g')
ylabel('Meditation');
xlabel('Time (s)');
legend('Meditation');
title('Meditation')
set(gca, 'fontweight', 'bold', 'fontsize', 16)
axis([0, 245, 32,100]);
subplot(2,2,4)
plot(magnitude)
ylabel('Magnitude');
xlabel('Hz');
legend('Attention');
title('Attention')
set(gca, 'fontweight', 'bold', 'fontsize', 16)
axis([0, length(Attention)/2, 0, 4000]);
%savinggraphs
filename=['sublot_', num2str(maximum)];
print(fullfile('C:\Users\Mmlab\Desktop\Graphs', filename), '-djpeg', '-r300');
